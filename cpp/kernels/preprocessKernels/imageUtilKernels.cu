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

#include "common/checkMacros.h"
#include "imageUtilKernels.h"
#include "kernels/common/vectorizedTypes.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace kernel
{

__global__ void normalizeImageKernel(unsigned char const* originalImage, float const* mean, float const* std,
    half* normalizedImage, int64_t const batch, int64_t const height, int64_t const width, int64_t const channels)
{
    // Each thread processes one pixel
    // originalImage format: [batch, height, width, channels]
    // normalizedImage format: [batch, height, width, channels]
    int64_t const tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const totalPixels = batch * height * width * channels;
    if (tid >= totalPixels)
        return;

    auto const channel = tid % channels;
    unsigned char val = originalImage[tid];
    float normalized = (val / 255.0f - mean[channel]) / std[channel];
    normalizedImage[tid] = __float2half(normalized);
}

void normalizeImage(rt::Tensor const& originalImage, rt::Tensor const& mean, rt::Tensor const& std,
    rt::Tensor& normalizedImage, cudaStream_t stream)
{
    check::check(originalImage.getDeviceType() == rt::DeviceType::kGPU && mean.getDeviceType() == rt::DeviceType::kGPU
            && std.getDeviceType() == rt::DeviceType::kGPU && normalizedImage.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(originalImage.getDataType() == DataType::kUINT8 && mean.getDataType() == DataType::kFLOAT
            && std.getDataType() == DataType::kFLOAT && normalizedImage.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(originalImage.getShape().getNumDims() == 4 && normalizedImage.getShape().getNumDims() == 4,
        "Input and output tensor shapes shall be [batch, height, width, channels] and [batch, height, width, channels] "
        "respectively.");

    int64_t const batch = originalImage.getShape()[0];
    int64_t const height = originalImage.getShape()[1];
    int64_t const width = originalImage.getShape()[2];
    int64_t const channels = originalImage.getShape()[3];
    int64_t const totalPixels = batch * height * width * channels;
    check::check(
        channels == mean.getShape()[0] && channels == std.getShape()[0], "Channels mismatch for mean and std.");

    // Each CTA get assigned 256 threads.
    uint32_t const blockSize = 256;
    uint32_t const gridSize = static_cast<uint32_t>((totalPixels + blockSize - 1) / blockSize);

    normalizeImageKernel<<<gridSize, blockSize, 0, stream>>>(originalImage.dataPointer<unsigned char>(),
        mean.dataPointer<float>(), std.dataPointer<float>(), normalizedImage.dataPointer<half>(), batch, height, width,
        channels);
}

__global__ void transposeToPatchQwenKernel(half const* originalImage, half* inputPatches, int64_t const T,
    int64_t const H, int64_t const W, int64_t const C, int64_t const temporalPatchSize, int64_t const patchSize,
    int64_t const mergeSize, int64_t const inputOffset)
{
    // This is a naive implementation of 9D transpose.
    // Each CTA get assigned 256 threads. Each thread processes one element
    // Original image format: [T, H, W, C]
    //      T = gridT * temporalPatchSize
    //      H = gridH * mergeSize * patchSize
    //      W = gridW * mergeSize * patchSize
    //      C = channels
    // Transposed format: [seqLength, inputDim]
    //      seqLength = gridT * gridH * gridW * mergeSize * mergeSize
    //      inputDim = C * temporalPatchSize * patchSize * patchSize

    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;

    auto const gridT = T / temporalPatchSize;
    auto const gridH = H / (mergeSize * patchSize);
    auto const gridW = W / (mergeSize * patchSize);
    auto const seqLength = gridT * gridH * gridW * mergeSize * mergeSize;
    auto const inputDim = C * temporalPatchSize * patchSize * patchSize;
    auto const totalElements = seqLength * inputDim;

    if (tid >= totalElements)
        return;

    // Calculate which sequence and element this thread handles
    auto const seqIdx = tid / inputDim;
    auto const elemIdx = tid % inputDim;

    // Calculate sequence coordinates
    auto const tIdx = seqIdx / (gridH * gridW * mergeSize * mergeSize);
    auto const hIdx = (seqIdx % (gridH * gridW * mergeSize * mergeSize)) / (gridW * mergeSize * mergeSize);
    auto const wIdx = (seqIdx % (gridW * mergeSize * mergeSize)) / (mergeSize * mergeSize);
    auto const mergeH = (seqIdx % (mergeSize * mergeSize)) / mergeSize;
    auto const mergeW = seqIdx % mergeSize;

    // Calculate coordinates within the patch
    auto const cIdx = elemIdx / (temporalPatchSize * patchSize * patchSize);
    auto const tPatchIdx = (elemIdx % (temporalPatchSize * patchSize * patchSize)) / (patchSize * patchSize);
    auto const patchH = (elemIdx % (patchSize * patchSize)) / patchSize;
    auto const patchW = elemIdx % patchSize;

    // Calculate source coordinates
    auto const srcT = tIdx * temporalPatchSize + tPatchIdx;
    auto const srcH = hIdx * mergeSize * patchSize + mergeH * patchSize + patchH;
    auto const srcW = wIdx * mergeSize * patchSize + mergeW * patchSize + patchW;
    auto const srcC = cIdx;

    // Calculate indices
    auto const srcIdx = srcT * H * W * C + srcH * W * C + srcW * C + srcC;
    auto const dstIdx = inputOffset + seqIdx * inputDim + elemIdx;

    // Direct copy (coalesced write, strided read)
    inputPatches[dstIdx] = originalImage[srcIdx];
}

void transposeToPatchQwenViT(rt::Tensor const& originalImage, rt::Tensor& inputPatches, int64_t const inputOffset,
    int64_t const temporalPatchSize, int64_t const patchSize, int64_t const mergeSize, cudaStream_t stream)
{
    check::check(
        originalImage.getDeviceType() == rt::DeviceType::kGPU && inputPatches.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(originalImage.getDataType() == DataType::kHALF && inputPatches.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(originalImage.getShape().getNumDims() == 4 && inputPatches.getShape().getNumDims() == 2,
        "Input and output tensor shapes shall be [T, H, W, C] and [totalSeqLength, inputDim] respectively.");
    // Get tensor dimensions
    int64_t const T = originalImage.getShape()[0];
    int64_t const H = originalImage.getShape()[1];
    int64_t const W = originalImage.getShape()[2];
    int64_t const C = originalImage.getShape()[3];
    int64_t const inputDim = inputPatches.getShape()[1];
    int64_t const totalElements = T * H * W * C;

    // Assertions for dimension assumptions
    check::check(inputDim == C * temporalPatchSize * patchSize * patchSize,
        "inputDim must be equal to C * temporalPatchSize * patchSize * patchSize: inputDim=" + std::to_string(inputDim)
            + ", C * temporalPatchSize * patchSize * patchSize="
            + std::to_string(C * temporalPatchSize * patchSize * patchSize));
    check::check(T % temporalPatchSize == 0,
        "T must be multiple of temporalPatchSize: T=" + std::to_string(T)
            + ", temporalPatchSize=" + std::to_string(temporalPatchSize));
    check::check(H % (mergeSize * patchSize) == 0,
        "H must be multiple of mergeSize * patchSize: H=" + std::to_string(H)
            + ", mergeSize * patchSize=" + std::to_string(mergeSize * patchSize));
    check::check(W % (mergeSize * patchSize) == 0,
        "W must be multiple of mergeSize * patchSize: W=" + std::to_string(W)
            + ", mergeSize * patchSize=" + std::to_string(mergeSize * patchSize));

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    transposeToPatchQwenKernel<<<gridSize, blockSize, 0, stream>>>(originalImage.dataPointer<half>(),
        inputPatches.dataPointer<half>(), T, H, W, C, temporalPatchSize, patchSize, mergeSize, inputOffset);
}

__global__ void transposeToPatchInternVLPhi4MMKernel(half const* originalImage, half* inputPatches,
    int64_t const inputOffset, int64_t const height, int64_t const width, int64_t const channels,
    int64_t const blockImageSizeH, int64_t const blockImageSizeW)
{
    // This is a naive implementation of 5D transpose.
    // Each CTA get assigned 256 threads. Each thread processes one element
    // Original image format: [1, H, W, C]
    //      H = gridH * blockImageSizeH
    //      W = gridW * blockImageSizeW
    //      C = channels
    // Transposed format: [gridH * gridW, channels, blockSizeH, blockSizeW]

    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto const gridH = height / blockImageSizeH;
    auto const gridW = width / blockImageSizeW;
    auto const numBlocks = gridH * gridW;
    auto const totalElements = numBlocks * channels * blockImageSizeH * blockImageSizeW;

    if (tid >= totalElements)
        return;

    // Calculate indices
    auto const gridHIdx = tid / (gridW * channels * blockImageSizeH * blockImageSizeW);
    auto const gridWIdx = (tid % (gridW * channels * blockImageSizeH * blockImageSizeW))
        / (channels * blockImageSizeH * blockImageSizeW);
    auto const cIdx = (tid % (channels * blockImageSizeH * blockImageSizeW)) / (blockImageSizeH * blockImageSizeW);
    auto const blockHIdx = (tid % (blockImageSizeH * blockImageSizeW)) / blockImageSizeW;
    auto const blockWIdx = tid % blockImageSizeW;

    auto const srcHIdx = gridHIdx * blockImageSizeH + blockHIdx;
    auto const srcWIdx = gridWIdx * blockImageSizeW + blockWIdx;
    auto const srcCIdx = cIdx;
    auto const srcIdx = srcHIdx * width * channels + srcWIdx * channels + srcCIdx;
    auto const dstIdx = inputOffset + tid;

    // Direct copy (coalesced write, strided read)
    inputPatches[dstIdx] = originalImage[srcIdx];
}

void transposeToPatchInternVLPhi4MM(
    rt::Tensor const& originalImage, rt::Tensor& inputPatches, int64_t const inputOffset, cudaStream_t stream)
{
    check::check(
        originalImage.getDeviceType() == rt::DeviceType::kGPU && inputPatches.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(originalImage.getDataType() == DataType::kHALF && inputPatches.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(originalImage.getShape().getNumDims() == 4 && inputPatches.getShape().getNumDims() == 4,
        "Input and output tensor shapes shall be [1, height, width, channels] and [totalNumBlocks, channels, "
        "blockSizeH, blockSizeW] respectively.");
    check::check(originalImage.getShape()[0] == 1, "Original image shape shall be [1, height, width, channels].");

    int64_t const height = originalImage.getShape()[1];
    int64_t const width = originalImage.getShape()[2];
    int64_t const channels = originalImage.getShape()[3];
    int64_t const blockSizeH = inputPatches.getShape()[2];
    int64_t const blockSizeW = inputPatches.getShape()[3];
    int64_t const totalElements = height * width * channels;

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    transposeToPatchInternVLPhi4MMKernel<<<gridSize, blockSize, 0, stream>>>(originalImage.dataPointer<half>(),
        inputPatches.dataPointer<half>(), inputOffset, height, width, channels, blockSizeH, blockSizeW);
}

// Phi4MM Pack All Batched Kernel
namespace
{
// copyHiddenVec
// Purpose:
//   Efficiently copy one token vector of length `hidden` (FP16) from src → dst.
//   Uses vectorized loads/stores (kernel::DVec<half>) to maximize memory
//   throughput on the hidden dimension which is contiguous in memory.
//
// Execution model:
//   - All threads in the CTA cooperate to copy one token:
//       each thread handles chunks in a round-robin fashion with stride = blockDim.x.
//   - Vector width V is chosen by DVec<half>::vec_size (typically 8 halves).
//   - Tail elements (< V) are copied by thread 0 to avoid race conditions.
//
// Rationale:
//   In all pack kernels below, we write tokens consecutively in the output,
//   making writes coalesced. Reads are strided because different tokens are
//   gathered, so vectorizing the hidden copy minimizes the cost of those reads.
__device__ __forceinline__ void copyHiddenVec(
    half const* __restrict__ src, half* __restrict__ dst, int32_t hidden, int32_t threadStride, int32_t threadIdxX)
{
    // Vectorized copy in chunks of 8 halves
    constexpr int32_t V = kernel::DVec<half>::vec_size;
    int32_t const numChunks = hidden / V;
    for (int32_t chunk = threadIdxX; chunk < numChunks; chunk += threadStride)
    {
        kernel::DVec<half> vec;
        vec.load(src + chunk * V);
        vec.store(dst + chunk * V);
    }
    // Tail copy by thread 0
    int32_t const tail = hidden % V;
    if (threadIdxX == 0)
    {
        int32_t const base = numChunks * V;
        for (int32_t i = 0; i < tail; ++i)
        {
            dst[base + i] = src[base + i];
        }
    }
}

// binarySearchImage
// Purpose:
//   Given a global output token index `tokenIdx` and an array `outStart` of size
//   `numImages` where each element denotes the starting output token offset of
//   an image, find the image index that owns `tokenIdx`.
//
// Contract/assumptions:
//   - outStart is monotonically non-decreasing.
//   - The i-th image covers output indices in [outStart[i], outStart[i+1]) for i < numImages-1,
//     and [outStart[numImages-1], totalOutTokens) for the last image.
//
// Returns:
//   The greatest index i such that outStart[i] <= tokenIdx.
__device__ __forceinline__ int32_t binarySearchImage(
    int64_t const* __restrict__ outStart, int32_t numImages, int64_t tokenIdx)
{
    int32_t lo = 0;
    int32_t hi = numImages - 1;
    int32_t ans = numImages - 1;
    while (lo <= hi)
    {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (outStart[mid] <= tokenIdx)
        {
            ans = mid;
            lo = mid + 1;
        }
        else
        {
            hi = mid - 1;
        }
    }
    return ans;
}
} // namespace

__global__ void phi4mmPostprocessVisionTokensKernel(
    half const* __restrict__ src, half* __restrict__ dst, Phi4MMIndex idx, Phi4MMGN gn)
{
    int64_t tokenIdx = static_cast<int64_t>(blockIdx.x);
    if (tokenIdx >= idx.totalOutTokens)
    {
        return;
    }

    int32_t const img = binarySearchImage(idx.dstOutStart, idx.numImages, tokenIdx);
    int64_t const localIdx = tokenIdx - idx.dstOutStart[img];

    int64_t const subLen = idx.subOutLen[img];

    half* dstPtr = dst + tokenIdx * idx.hidden;

    if (localIdx < subLen)
    {
        // sub segment
        int32_t const wb = idx.wBlocks[img];
        int64_t const cols = kTokensPerSidePhi4 * wb;
        int64_t const strideOut = cols + 1;
        int64_t const r = localIdx / strideOut;
        int64_t const c = localIdx % strideOut;
        if (c == cols)
        {
            copyHiddenVec(gn.subGN, dstPtr, idx.hidden, blockDim.x, threadIdx.x);
            return;
        }
        int64_t const bRow = r / kTokensPerSidePhi4;
        int64_t const pRow = r % kTokensPerSidePhi4;
        int64_t const bCol = c / kTokensPerSidePhi4;
        int64_t const pCol = c % kTokensPerSidePhi4;
        int64_t const blockId = bRow * wb + bCol;
        int64_t const patchId = pRow * kTokensPerSidePhi4 + pCol;
        int64_t const srcTokIndex = idx.srcSubStart[img] + blockId * kTokensPerBlockPhi4 + patchId;
        half const* srcPtr = src + srcTokIndex * idx.hidden;
        copyHiddenVec(srcPtr, dstPtr, idx.hidden, blockDim.x, threadIdx.x);
        return;
    }
    else if (localIdx == subLen)
    {
        // single glb_GN
        copyHiddenVec(gn.glbGN, dstPtr, idx.hidden, blockDim.x, threadIdx.x);
        return;
    }
    else
    {
        // glb segment
        int64_t const idx2 = localIdx - (subLen + 1);
        int64_t const cols = kTokensPerSidePhi4;
        int64_t const strideOut = cols + 1;
        int64_t const r = idx2 / strideOut;
        int64_t const c = idx2 % strideOut;
        if (c == cols)
        {
            copyHiddenVec(gn.subGN, dstPtr, idx.hidden, blockDim.x, threadIdx.x);
            return;
        }
        int64_t const srcTokIndex = idx.srcGlbStart[img] + r * kTokensPerSidePhi4 + c;
        half const* srcPtr = src + srcTokIndex * idx.hidden;
        copyHiddenVec(srcPtr, dstPtr, idx.hidden, blockDim.x, threadIdx.x);
        return;
    }
}

void phi4mmPostprocessVisionTokens(rt::Tensor const& srcEmbedding, rt::Tensor& dstEmbedding, Phi4MMIndex const& indices,
    Phi4MMGN const& gn, int64_t totalOutTokens, cudaStream_t stream)
{
    check::check(
        srcEmbedding.getDeviceType() == rt::DeviceType::kGPU && dstEmbedding.getDeviceType() == rt::DeviceType::kGPU,
        "phi4mmPostprocessVisionTokens(): All tensors must be on GPU.");
    check::check(srcEmbedding.getDataType() == DataType::kHALF && dstEmbedding.getDataType() == DataType::kHALF,
        "phi4mmPostprocessVisionTokens(): Embeddings and dstEmbedding must be FP16.");

    int32_t const hidden = static_cast<int32_t>(srcEmbedding.getShape()[1]);
    check::check(hidden == dstEmbedding.getShape()[1],
        "phi4mmPostprocessVisionTokens(): srcEmbedding and dstEmbedding must have the same hidden size.");

    // Require enough space for totalOutTokens * hidden elements of dstEmbedding's data type.
    int64_t const bytesPerElem = static_cast<int64_t>(rt::utils::getTypeSize(dstEmbedding.getDataType()));
    int64_t const requiredBytes = totalOutTokens * static_cast<int64_t>(hidden) * bytesPerElem;
    check::check(requiredBytes <= dstEmbedding.getMemoryCapacity(),
        "phi4mmPostprocessVisionTokens(): Total output tokens exceed dstEmbedding memory capacity.");

    dim3 block(128);
    dim3 grid(static_cast<uint32_t>(totalOutTokens));
    phi4mmPostprocessVisionTokensKernel<<<grid, block, 0, stream>>>(
        srcEmbedding.dataPointer<half>(), dstEmbedding.dataPointer<half>(), indices, gn);
}

__global__ void initMaskToMinKernel(half* attentionMask, int32_t const totalElements)
{
    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalElements)
    {
        return;
    }

    // Mask used to disable attention between "patches". Use -20000.0F to avoid data overflow in
    // TensorRT which could produce NaN output if we supply -MAX_FLOAT_FP16 mask values.
    half const disabledMaskValue{-20000.0F};
    attentionMask[tid] = disabledMaskValue;
}

__global__ void initAttentionMaskKernel(
    int64_t const* cuSeqlens, half* attentionMask, int64_t const cuSeqlensSize, int64_t const curHW)
{
    // Each CTA process one cuSeqlen and set zero to attentionMask[start:end, start:end]
    // Each CTA get assigned (16, 16) threads.
    auto const bIdx = blockIdx.x;
    auto const start = cuSeqlens[bIdx];
    auto const end = cuSeqlens[bIdx + 1];
    auto const tIdx = threadIdx.x;
    auto const tidy = threadIdx.y;

    for (auto i = start + tIdx; i < end; i += 16)
    {
        for (auto j = start + tidy; j < end; j += 16)
        {
            auto const posIdx = i * curHW + j;
            attentionMask[posIdx] = __float2half(0.0f);
        }
    }
}

void initAttentionMaskQwenViT(rt::Tensor const& cuSeqlens, rt::Tensor& attentionMask, cudaStream_t stream)
{
    check::check(
        cuSeqlens.getDeviceType() == rt::DeviceType::kGPU && attentionMask.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(cuSeqlens.getDataType() == DataType::kINT64 && attentionMask.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(cuSeqlens.getShape().getNumDims() == 1, "Cu seqlens shape shall be [num].");
    check::check(attentionMask.getShape().getNumDims() == 3 && attentionMask.getShape()[0] == 1,
        "Attention mask shape shall be [1, curHW, curHW].");

    int64_t const cuSeqlensSize = cuSeqlens.getShape()[0];
    int64_t const curHW = attentionMask.getShape()[1];
    int64_t const totalElements = curHW * curHW;

    // Initialize attention mask to small value to indicate "disabled" attention.
    uint32_t const initBlockSize = 256;
    uint32_t const initGridSize = (totalElements + initBlockSize - 1) / initBlockSize;
    initMaskToMinKernel<<<initGridSize, initBlockSize, 0, stream>>>(attentionMask.dataPointer<half>(), totalElements);

    // Set zero to target positions
    dim3 blockSize{16, 16};
    uint32_t const gridSize = cuSeqlensSize - 1;
    initAttentionMaskKernel<<<gridSize, blockSize, 0, stream>>>(
        cuSeqlens.dataPointer<int64_t>(), attentionMask.dataPointer<half>(), cuSeqlensSize, curHW);
}

__global__ void initRotaryPosEmbQwenKernel(float* rotaryPosEmb, int64_t const T, int64_t const H, int64_t const W,
    int64_t const mergeSize, int64_t const startIdx, int64_t const vitPosEmbDim, float const rotaryBaseFrequency,
    float const scale)
{
    // Each CTA get assigned 256 threads. Each thread processes one element
    // rotaryPosEmb: [totalSeqLength, vitPosEmbDim]
    //     [T, (llmGridH, llmGridW, mergeSize, mergeSize), (2, vitPosEmbDim/2)]
    //     where llmGridH = H / mergeSize, llmGridW = W / mergeSize and position ids is duplicated for T
    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto const totalElements = T * H * W * vitPosEmbDim;
    if (tid >= totalElements)
        return;

    auto const hwIdx = (tid % (H * W * vitPosEmbDim)) / (vitPosEmbDim);
    auto const hOrWPos = (tid % (vitPosEmbDim)) / (vitPosEmbDim / 2);
    auto const dimIdx = tid % (vitPosEmbDim / 2);

    int64_t const llmGridH = H / mergeSize;
    int64_t const llmGridW = W / mergeSize;
    auto const llmGridHIdx = hwIdx / (llmGridW * mergeSize * mergeSize);
    auto const llmGridWIdx = (hwIdx % (llmGridW * mergeSize * mergeSize)) / (mergeSize * mergeSize);
    auto const mergeHIdx = (hwIdx % (mergeSize * mergeSize)) / mergeSize;
    auto const mergeWIdx = hwIdx % mergeSize;

    auto const originalHIdx = llmGridHIdx * mergeSize + mergeHIdx;
    auto const originalWIdx = llmGridWIdx * mergeSize + mergeWIdx;
    // 0: H, 1: W
    auto const posId = (hOrWPos == 0) ? originalHIdx : originalWIdx;

    float invFreq = posId * scale / pow(rotaryBaseFrequency, 2 * dimIdx / (float) vitPosEmbDim);
    rotaryPosEmb[startIdx * vitPosEmbDim + tid] = invFreq;
}

void initRotaryPosEmbQwenViT(rt::Tensor& rotaryPosEmb, std::vector<int64_t> const& gridTHW, int64_t const mergeSize,
    int64_t const startIdx, float const rotaryBaseFrequency, float const scale, cudaStream_t stream)
{
    check::check(rotaryPosEmb.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall be GPU for the rotary position embeddings tensor.");
    check::check(rotaryPosEmb.getDataType() == DataType::kFLOAT,
        "Data type shall be float for the rotary position embeddings tensor.");
    check::check(rotaryPosEmb.getShape().getNumDims() == 2,
        "Rotary position embeddings shape shall be [totalSeqLength, vitPosEmbDim].");

    check::check(gridTHW.size() == 3, "gridTHW must have exactly 3 elements [T, H, W]");
    int64_t const T = gridTHW[0];
    int64_t const H = gridTHW[1];
    int64_t const W = gridTHW[2];

    int64_t const vitPosEmbDim = rotaryPosEmb.getShape()[1];
    int64_t const totalElements = T * H * W * vitPosEmbDim;

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    initRotaryPosEmbQwenKernel<<<gridSize, blockSize, 0, stream>>>(
        rotaryPosEmb.dataPointer<float>(), T, H, W, mergeSize, startIdx, vitPosEmbDim, rotaryBaseFrequency, scale);
}

__global__ void initFastPosEmbedQwenViTKernel(int64_t* fastPosEmbedIdx, half* fastPosEmbedWeight,
    int64_t const llmGridH, int64_t const llmGridW, int64_t const mergeSize, int64_t const numGridPerSide,
    float const lineSpaceH, float const lineSpaceW, int64_t const startIdx, int64_t const totalSeqLength)
{
    // Each CTA get assigned 256 threads. Each thread processes one position in grid
    //     [llmGridH, llmGridW, mergeSize, mergeSize]
    // Each position needs to generate 4 indices and 4 weights
    // fastPosEmbedIdx: [4, totalSeqLength]
    // fastPosEmbedWeight: [4, totalSeqLength]
    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto const totalElements = llmGridH * llmGridW * mergeSize * mergeSize;
    if (tid >= totalElements)
        return;

    auto const llmGridHIdx = tid / (llmGridW * mergeSize * mergeSize);
    auto const llmGridWIdx = (tid % (llmGridW * mergeSize * mergeSize)) / (mergeSize * mergeSize);
    auto const mergeHIdx = (tid % (mergeSize * mergeSize)) / mergeSize;
    auto const mergeWIdx = tid % mergeSize;

    float const hIdx = lineSpaceH * (llmGridHIdx * mergeSize + mergeHIdx);
    float const wIdx = lineSpaceW * (llmGridWIdx * mergeSize + mergeWIdx);

    int64_t const hIdxFloor = static_cast<int64_t>(hIdx);
    int64_t const wIdxFloor = static_cast<int64_t>(wIdx);
    int64_t const hIdxCeil = std::min(hIdxFloor + 1, (numGridPerSide - 1));
    int64_t const wIdxCeil = std::min(wIdxFloor + 1, (numGridPerSide - 1));

    float const dh = hIdx - hIdxFloor;
    float const dw = wIdx - wIdxFloor;

    int64_t const baseH = hIdxFloor * numGridPerSide;
    int64_t const baseHCeil = hIdxCeil * numGridPerSide;

    int64_t const targetIdx = startIdx + tid;

    fastPosEmbedIdx[0 * totalSeqLength + targetIdx] = baseH + wIdxFloor;
    fastPosEmbedIdx[1 * totalSeqLength + targetIdx] = baseH + wIdxCeil;
    fastPosEmbedIdx[2 * totalSeqLength + targetIdx] = baseHCeil + wIdxFloor;
    fastPosEmbedIdx[3 * totalSeqLength + targetIdx] = baseHCeil + wIdxCeil;
    fastPosEmbedWeight[0 * totalSeqLength + targetIdx] = __float2half((1 - dh) * (1 - dw));
    fastPosEmbedWeight[1 * totalSeqLength + targetIdx] = __float2half((1 - dh) * dw);
    fastPosEmbedWeight[2 * totalSeqLength + targetIdx] = __float2half(dh * (1 - dw));
    fastPosEmbedWeight[3 * totalSeqLength + targetIdx] = __float2half(dh * dw);
}

void initFastPosEmbedQwenViT(rt::Tensor& fastPosEmbedIdx, rt::Tensor& fastPosEmbedWeight,
    std::vector<int64_t> const& gridTHW, int64_t const mergeSize, int64_t const numGridPerSide, int64_t const startIdx,
    cudaStream_t stream)
{
    check::check(fastPosEmbedIdx.getDeviceType() == rt::DeviceType::kGPU
            && fastPosEmbedWeight.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(
        fastPosEmbedIdx.getDataType() == DataType::kINT64 && fastPosEmbedWeight.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(fastPosEmbedIdx.getShape().getNumDims() == 2 && fastPosEmbedIdx.getShape()[0] == 4,
        "Fast position embeddings index shapes shall be [4, totalSeqLength].");
    check::check(fastPosEmbedWeight.getShape().getNumDims() == 2 && fastPosEmbedWeight.getShape()[0] == 4,
        "Fast position embeddings weight shapes shall be [4, totalSeqLength].");

    int64_t const totalSeqLength = fastPosEmbedIdx.getShape()[1];
    check::check(totalSeqLength == fastPosEmbedWeight.getShape()[1], "Total sequence length mismatch.");

    check::check(gridTHW.size() == 3, "gridTHW must have exactly 3 elements [T, H, W]");
    int64_t const H = gridTHW[1];
    int64_t const W = gridTHW[2];
    int64_t const llmGridH = H / mergeSize;
    int64_t const llmGridW = W / mergeSize;
    float const lineSpaceH = static_cast<float>(numGridPerSide - 1) / (H - 1);
    float const lineSpaceW = static_cast<float>(numGridPerSide - 1) / (W - 1);

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (H * W + blockSize - 1) / blockSize;

    initFastPosEmbedQwenViTKernel<<<gridSize, blockSize, 0, stream>>>(fastPosEmbedIdx.dataPointer<int64_t>(),
        fastPosEmbedWeight.dataPointer<half>(), llmGridH, llmGridW, mergeSize, numGridPerSide, lineSpaceH, lineSpaceW,
        startIdx, totalSeqLength);
}

} // namespace kernel
} // namespace trt_edgellm
