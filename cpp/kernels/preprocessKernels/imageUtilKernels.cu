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

/* Fused image preprocessing adapted from
 * https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/7c1623f/libraries/YUVToRGB
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (c) NVIDIA CORPORATION & AFFILIATES
 */

#include "common/checkMacros.h"
#include "imageUtilKernels.h"
#include "kernels/common/vectorizedTypes.cuh"
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace kernel
{

__global__ void transposeToPatchQwenKernel(half const* originalImage, half* inputPatches, int64_t const T,
    int64_t const H, int64_t const W, int64_t const C, int64_t const temporalPatchSize, int64_t const patchSize,
    int64_t const mergeSize, bool const temporalFirst, bool const channelLast, int64_t const inputOffset)
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
    int64_t cIdx;
    int64_t tPatchIdx;
    int64_t patchH;
    int64_t patchW;
    if (channelLast)
    {
        patchH = elemIdx / (patchSize * C * temporalPatchSize);
        patchW = (elemIdx % (patchSize * C * temporalPatchSize)) / (C * temporalPatchSize);
        cIdx = (elemIdx % (C * temporalPatchSize)) / temporalPatchSize;
        tPatchIdx = elemIdx % temporalPatchSize;
    }
    else
    {
        cIdx = temporalFirst ? (elemIdx % (C * patchSize * patchSize)) / (patchSize * patchSize)
                             : elemIdx / (temporalPatchSize * patchSize * patchSize);
        tPatchIdx = temporalFirst ? elemIdx / (C * patchSize * patchSize)
                                  : (elemIdx % (temporalPatchSize * patchSize * patchSize)) / (patchSize * patchSize);
        patchH = (elemIdx % (patchSize * patchSize)) / patchSize;
        patchW = elemIdx % patchSize;
    }

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

__global__ void transposeToPatchGemma4Kernel(half const* originalImage, half* inputPatches, int64_t const H,
    int64_t const W, int64_t const C, int64_t const patchSize, int64_t const inputOffset)
{
    // Gemma4 patchification: channel-last within each patch
    // Original image format: [1, H, W, C]
    // Output format: [numPatches, patchSize * patchSize * C]
    //   where element order within each patch is [patchH, patchW, C] (channels fastest)
    // This matches HuggingFace Gemma4 convert_image_to_patches():
    //   image.reshape(C, pH, ps, pW, ps).permute(1,3,2,4,0).reshape(pH*pW, ps*ps*C)

    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;

    auto const gridH = H / patchSize;
    auto const gridW = W / patchSize;
    auto const numPatches = gridH * gridW;
    auto const inputDim = patchSize * patchSize * C;
    auto const totalElements = numPatches * inputDim;

    if (tid >= totalElements)
        return;

    // Calculate which patch and element within patch
    auto const patchIdx = tid / inputDim;
    auto const elemIdx = tid % inputDim;

    // Patch grid coordinates
    auto const hIdx = patchIdx / gridW;
    auto const wIdx = patchIdx % gridW;

    // Element coordinates within patch: [patchH, patchW, C] ordering
    auto const patchH = elemIdx / (patchSize * C);
    auto const patchW = (elemIdx % (patchSize * C)) / C;
    auto const cIdx = elemIdx % C;

    // Source coordinates in [H, W, C] image
    auto const srcH = hIdx * patchSize + patchH;
    auto const srcW = wIdx * patchSize + patchW;
    auto const srcIdx = srcH * W * C + srcW * C + cIdx;
    auto const dstIdx = inputOffset + tid;

    inputPatches[dstIdx] = originalImage[srcIdx];
}

void transposeToPatchGemma4ViT(rt::Tensor const& originalImage, rt::Tensor& inputPatches, int64_t const inputOffset,
    int64_t const patchSize, cudaStream_t stream)
{
    check::check(
        originalImage.getDeviceType() == rt::DeviceType::kGPU && inputPatches.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(originalImage.getDataType() == DataType::kHALF && inputPatches.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(originalImage.getShape().getNumDims() == 4 && inputPatches.getShape().getNumDims() == 2,
        "Input and output tensor shapes shall be [1, H, W, C] and [totalSeqLength, inputDim] respectively.");

    int64_t const H = originalImage.getShape()[1];
    int64_t const W = originalImage.getShape()[2];
    int64_t const C = originalImage.getShape()[3];
    int64_t const inputDim = inputPatches.getShape()[1];

    check::check(inputDim == patchSize * patchSize * C,
        "inputDim must equal patchSize * patchSize * C: inputDim=" + std::to_string(inputDim)
            + ", patchSize*patchSize*C=" + std::to_string(patchSize * patchSize * C));
    check::check(H % patchSize == 0 && W % patchSize == 0, "H and W must be multiples of patchSize");

    int64_t const gridH = H / patchSize;
    int64_t const gridW = W / patchSize;
    int64_t const totalElements = gridH * gridW * inputDim;
    check::check(inputOffset >= 0 && inputOffset + totalElements <= inputPatches.getShape().volume(),
        "inputOffset + totalElements must fit inside inputPatches: inputOffset=" + std::to_string(inputOffset)
            + ", totalElements=" + std::to_string(totalElements)
            + ", capacity=" + std::to_string(inputPatches.getShape().volume()));

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    transposeToPatchGemma4Kernel<<<gridSize, blockSize, 0, stream>>>(
        originalImage.dataPointer<half>(), inputPatches.dataPointer<half>(), H, W, C, patchSize, inputOffset);
}

void transposeToPatchQwenViT(rt::Tensor const& originalImage, rt::Tensor& inputPatches, int64_t const inputOffset,
    int64_t const temporalPatchSize, int64_t const patchSize, int64_t const mergeSize, bool const temporalFirst,
    bool const channelLast, cudaStream_t stream)
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
    check::check(!(temporalFirst && channelLast), "temporalFirst and channelLast patch layouts are mutually exclusive");
    check::check(H % (mergeSize * patchSize) == 0,
        "H must be multiple of mergeSize * patchSize: H=" + std::to_string(H)
            + ", mergeSize * patchSize=" + std::to_string(mergeSize * patchSize));
    check::check(W % (mergeSize * patchSize) == 0,
        "W must be multiple of mergeSize * patchSize: W=" + std::to_string(W)
            + ", mergeSize * patchSize=" + std::to_string(mergeSize * patchSize));

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    transposeToPatchQwenKernel<<<gridSize, blockSize, 0, stream>>>(originalImage.dataPointer<half>(),
        inputPatches.dataPointer<half>(), T, H, W, C, temporalPatchSize, patchSize, mergeSize, temporalFirst,
        channelLast, inputOffset);
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

__global__ void initRotaryPosEmbGemma4Kernel(float* rotaryPosEmb, int64_t const* pixelPositionIds,
    int64_t const totalSeqLength, int64_t const headDim, float const rotaryBaseFrequency)
{
    int64_t const tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const totalElements = totalSeqLength * headDim;
    if (tid >= totalElements)
        return;

    int64_t const tokenIdx = tid / headDim;
    int64_t const dimIdx = tid % headDim;
    int64_t const axisDim = headDim / 2;
    int64_t const axis = dimIdx / axisDim;
    int64_t const dimInAxis = dimIdx % axisDim;
    int64_t const freqIdx = dimInAxis % (axisDim / 2);
    int64_t const posId = pixelPositionIds[tokenIdx * 2 + axis];

    float const exponent = 2.0F * static_cast<float>(freqIdx) / static_cast<float>(axisDim);
    rotaryPosEmb[tid] = static_cast<float>(posId) / powf(rotaryBaseFrequency, exponent);
}

void initRotaryPosEmbGemma4ViT(
    rt::Tensor& rotaryPosEmb, rt::Tensor const& pixelPositionIds, float rotaryBaseFrequency, cudaStream_t stream)
{
    check::check(rotaryPosEmb.getDeviceType() == rt::DeviceType::kGPU
            && pixelPositionIds.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall be GPU for Gemma4 rotary position tensors.");
    check::check(rotaryPosEmb.getDataType() == DataType::kFLOAT && pixelPositionIds.getDataType() == DataType::kINT64,
        "Data type check failed for Gemma4 rotary position tensors.");
    check::check(rotaryPosEmb.getShape().getNumDims() == 2,
        "Gemma4 rotary position embeddings shape shall be [totalSeqLength, headDim].");
    check::check(pixelPositionIds.getShape().getNumDims() == 2 && pixelPositionIds.getShape()[1] == 2,
        "Gemma4 pixel position ids shape shall be [totalSeqLength, 2].");
    check::check(rotaryPosEmb.getShape()[0] == pixelPositionIds.getShape()[0],
        "Gemma4 rotary position embeddings and pixel position ids must have the same sequence length.");

    int64_t const totalSeqLength = rotaryPosEmb.getShape()[0];
    int64_t const headDim = rotaryPosEmb.getShape()[1];
    check::check(headDim % 4 == 0, "Gemma4 RoPE headDim must be divisible by 4.");

    uint32_t const blockSize = 256;
    uint32_t const gridSize = static_cast<uint32_t>((totalSeqLength * headDim + blockSize - 1) / blockSize);
    initRotaryPosEmbGemma4Kernel<<<gridSize, blockSize, 0, stream>>>(rotaryPosEmb.dataPointer<float>(),
        pixelPositionIds.dataPointer<int64_t>(), totalSeqLength, headDim, rotaryBaseFrequency);
}

__global__ void initPoolingWeightsGemma4Kernel(half* poolingWeights, int64_t const totalPatches,
    int64_t const patchStart, int64_t const softStart, int64_t const patchHeight, int64_t const patchWidth,
    int64_t const poolingKernelSize)
{
    int64_t const tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const curPatches = patchHeight * patchWidth;
    if (tid >= curPatches)
        return;

    int64_t const y = tid / patchWidth;
    int64_t const x = tid % patchWidth;
    int64_t const softWidth = patchWidth / poolingKernelSize;
    int64_t const row = softStart + (y / poolingKernelSize) * softWidth + (x / poolingKernelSize);
    int64_t const col = patchStart + tid;
    float const weight = 1.0F / static_cast<float>(poolingKernelSize * poolingKernelSize);
    poolingWeights[row * totalPatches + col] = __float2half(weight);
}

void initPoolingWeightsGemma4ViT(rt::Tensor& poolingWeights, int64_t patchStart, int64_t softStart, int64_t patchHeight,
    int64_t patchWidth, int64_t poolingKernelSize, cudaStream_t stream)
{
    check::check(poolingWeights.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall be GPU for Gemma4 pooling weights tensor.");
    check::check(
        poolingWeights.getDataType() == DataType::kHALF, "Data type shall be half for Gemma4 pooling weights tensor.");
    check::check(poolingWeights.getShape().getNumDims() == 2,
        "Gemma4 pooling weights shape shall be [totalSoftTokens, totalPatches].");
    check::check(poolingKernelSize > 0, "Gemma4 pooling kernel size must be positive.");
    check::check(patchHeight > 0 && patchWidth > 0, "Gemma4 patch grid size must be positive.");
    check::check(patchHeight % poolingKernelSize == 0 && patchWidth % poolingKernelSize == 0,
        "Gemma4 patch grid size must be divisible by pooling kernel size.");

    int64_t const totalSoftTokens = poolingWeights.getShape()[0];
    int64_t const totalPatches = poolingWeights.getShape()[1];
    int64_t const curPatches = patchHeight * patchWidth;
    int64_t const curSoftTokens = (patchHeight / poolingKernelSize) * (patchWidth / poolingKernelSize);
    check::check(patchStart >= 0 && patchStart + curPatches <= totalPatches,
        "Gemma4 pooling patch range exceeds pooling weights shape.");
    check::check(softStart >= 0 && softStart + curSoftTokens <= totalSoftTokens,
        "Gemma4 pooling soft-token range exceeds pooling weights shape.");

    uint32_t const blockSize = 256;
    uint32_t const gridSize = static_cast<uint32_t>((curPatches + blockSize - 1) / blockSize);
    initPoolingWeightsGemma4Kernel<<<gridSize, blockSize, 0, stream>>>(poolingWeights.dataPointer<half>(), totalPatches,
        patchStart, softStart, patchHeight, patchWidth, poolingKernelSize);
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
    int64_t const T = gridTHW[0];
    int64_t const H = gridTHW[1];
    int64_t const W = gridTHW[2];
    int64_t const llmGridH = H / mergeSize;
    int64_t const llmGridW = W / mergeSize;
    float const lineSpaceH = static_cast<float>(numGridPerSide - 1) / (H - 1);
    float const lineSpaceW = static_cast<float>(numGridPerSide - 1) / (W - 1);

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (H * W + blockSize - 1) / blockSize;

    // The fast position embedding is spatial (H*W); for a video grid it repeats per temporal frame (HF
    // `pos_embed.repeat(t, 1)`). Qwen3-VL splits frames into T=1 sub-span grids (this loop runs once);
    // Qwen3-Omni passes a single (T, H, W) grid, so write the same spatial pattern at each frame's patch offset.
    for (int64_t t = 0; t < T; ++t)
    {
        initFastPosEmbedQwenViTKernel<<<gridSize, blockSize, 0, stream>>>(fastPosEmbedIdx.dataPointer<int64_t>(),
            fastPosEmbedWeight.dataPointer<half>(), llmGridH, llmGridW, mergeSize, numGridPerSide, lineSpaceH,
            lineSpaceW, startIdx + t * H * W, totalSeqLength);
    }
}

__global__ void initFastPosEmbedCosmos3ViTKernel(int64_t* fastPosEmbedIdx, half* fastPosEmbedWeight, int64_t const H,
    int64_t const W, int64_t const mergeSize, int64_t const numGridPerSide, int64_t const startIdx,
    int64_t const totalSeqLength)
{
    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto const totalElements = H * W;
    if (tid >= totalElements)
        return;

    int64_t const llmGridW = W / mergeSize;
    auto const llmGridHIdx = tid / (llmGridW * mergeSize * mergeSize);
    auto const llmGridWIdx = (tid % (llmGridW * mergeSize * mergeSize)) / (mergeSize * mergeSize);
    auto const mergeHIdx = (tid % (mergeSize * mergeSize)) / mergeSize;
    auto const mergeWIdx = tid % mergeSize;

    int64_t const targetH = llmGridHIdx * mergeSize + mergeHIdx;
    int64_t const targetW = llmGridWIdx * mergeSize + mergeWIdx;

    // PyTorch interpolate(..., mode="bilinear", align_corners=false) uses half-pixel source
    // coordinates. Clamp before forming the two neighbors to reproduce border replication.
    float const hIdx = fminf(fmaxf((static_cast<float>(targetH) + 0.5F) * numGridPerSide / H - 0.5F, 0.0F),
        static_cast<float>(numGridPerSide - 1));
    float const wIdx = fminf(fmaxf((static_cast<float>(targetW) + 0.5F) * numGridPerSide / W - 0.5F, 0.0F),
        static_cast<float>(numGridPerSide - 1));

    int64_t const hIdxFloor = static_cast<int64_t>(floorf(hIdx));
    int64_t const wIdxFloor = static_cast<int64_t>(floorf(wIdx));
    int64_t const hIdxCeil = min(hIdxFloor + 1, numGridPerSide - 1);
    int64_t const wIdxCeil = min(wIdxFloor + 1, numGridPerSide - 1);
    float const dh = hIdx - hIdxFloor;
    float const dw = wIdx - wIdxFloor;

    int64_t const targetIdx = startIdx + tid;
    int64_t const baseH = hIdxFloor * numGridPerSide;
    int64_t const baseHCeil = hIdxCeil * numGridPerSide;
    fastPosEmbedIdx[0 * totalSeqLength + targetIdx] = baseH + wIdxFloor;
    fastPosEmbedIdx[1 * totalSeqLength + targetIdx] = baseH + wIdxCeil;
    fastPosEmbedIdx[2 * totalSeqLength + targetIdx] = baseHCeil + wIdxFloor;
    fastPosEmbedIdx[3 * totalSeqLength + targetIdx] = baseHCeil + wIdxCeil;
    fastPosEmbedWeight[0 * totalSeqLength + targetIdx] = __float2half((1.0F - dh) * (1.0F - dw));
    fastPosEmbedWeight[1 * totalSeqLength + targetIdx] = __float2half((1.0F - dh) * dw);
    fastPosEmbedWeight[2 * totalSeqLength + targetIdx] = __float2half(dh * (1.0F - dw));
    fastPosEmbedWeight[3 * totalSeqLength + targetIdx] = __float2half(dh * dw);
}

void initFastPosEmbedCosmos3ViT(rt::Tensor& fastPosEmbedIdx, rt::Tensor& fastPosEmbedWeight,
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
    check::check(gridTHW.size() == 3, "gridTHW must have exactly 3 elements [T, H, W]");

    int64_t const totalSeqLength = fastPosEmbedIdx.getShape()[1];
    check::check(totalSeqLength == fastPosEmbedWeight.getShape()[1], "Total sequence length mismatch.");
    int64_t const T = gridTHW[0];
    int64_t const H = gridTHW[1];
    int64_t const W = gridTHW[2];
    check::check(T > 0 && H > 0 && W > 0 && mergeSize > 0 && numGridPerSide > 0,
        "Cosmos3 fast position embedding dimensions must be positive.");
    check::check(H % mergeSize == 0 && W % mergeSize == 0,
        "Cosmos3 fast position embedding grid must be divisible by mergeSize.");
    check::check(startIdx >= 0 && startIdx + T * H * W <= totalSeqLength,
        "Cosmos3 fast position embedding range exceeds the output tensors.");

    uint32_t constexpr blockSize = 256;
    uint32_t const gridSize = static_cast<uint32_t>((H * W + blockSize - 1) / blockSize);
    for (int64_t t = 0; t < T; ++t)
    {
        initFastPosEmbedCosmos3ViTKernel<<<gridSize, blockSize, 0, stream>>>(fastPosEmbedIdx.dataPointer<int64_t>(),
            fastPosEmbedWeight.dataPointer<half>(), H, W, mergeSize, numGridPerSide, startIdx + t * H * W,
            totalSeqLength);
    }
}

__global__ void initRotaryPosEmbMuseGlimmerKernel(float* rotaryPosEmb, int64_t const T, int64_t const H,
    int64_t const W, int64_t const startIdx, int64_t const vitPosEmbDim, float const rotaryBaseFrequency)
{
    // Each thread processes one element of rotaryPosEmb: [totalSeqLength, vitPosEmbDim]
    //     [T, (H, W), (freq_w[vitPosEmbDim/2], freq_h[vitPosEmbDim/2])]
    // Tokens are in raster order (spatial_merge_size == 1); positions are offset by +1 (mirrors the
    // reference position_ids.flip(-1) + 1). inv_freq[k] = 1 / theta^(2*k / vitPosEmbDim).
    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto const totalElements = T * H * W * vitPosEmbDim;
    if (tid >= totalElements)
        return;

    auto const half = vitPosEmbDim / 2;
    // Raster token position within a frame (T frames repeat the same spatial pattern).
    auto const hwIdx = (tid / vitPosEmbDim) % (H * W);
    auto const hIdx = hwIdx / W;
    auto const wIdx = hwIdx % W;

    auto const dimIdx = tid % vitPosEmbDim;
    bool const isH = dimIdx >= half; // first half is freq_w, second half is freq_h
    auto const freqIdx = dimIdx % half;
    // +1 position offset folded in here.
    int64_t const posId = isH ? (hIdx + 1) : (wIdx + 1);

    float const exponent = 2.0f * static_cast<float>(freqIdx) / static_cast<float>(vitPosEmbDim);
    rotaryPosEmb[startIdx * vitPosEmbDim + tid] = static_cast<float>(posId) / powf(rotaryBaseFrequency, exponent);
}

void initRotaryPosEmbMuseGlimmerViT(rt::Tensor& rotaryPosEmb, std::vector<int64_t> const& gridTHW,
    int64_t const startIdx, float const rotaryBaseFrequency, cudaStream_t stream)
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
    check::check(vitPosEmbDim % 2 == 0, "Muse-Glimmer vitPosEmbDim must be even (concat(freq_w, freq_h)).");
    int64_t const totalElements = T * H * W * vitPosEmbDim;

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    initRotaryPosEmbMuseGlimmerKernel<<<gridSize, blockSize, 0, stream>>>(
        rotaryPosEmb.dataPointer<float>(), T, H, W, startIdx, vitPosEmbDim, rotaryBaseFrequency);
}

__global__ void initFastPosEmbedMuseGlimmerKernel(int64_t* fastPosEmbedIdx, half* fastPosEmbedWeight, int64_t const H,
    int64_t const W, int64_t const numGridPerSide, float const sideOverH, float const sideOverW, int64_t const startIdx,
    int64_t const totalSeqLength)
{
    // Each thread processes one raster token (spatial_merge_size == 1) and emits its 4 bilinear taps.
    // align_corners == False sampling of the numGridPerSide x numGridPerSide learned position table with
    // "zeros" padding: out-of-range taps are index-clamped but contribute zero weight.
    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto const totalElements = H * W;
    if (tid >= totalElements)
        return;

    auto const hIdx = tid / W;
    auto const wIdx = tid % W;

    // grid coordinate = (i + 0.5) * (side / dim) - 0.5
    float const hGrid = (static_cast<float>(hIdx) + 0.5f) * sideOverH - 0.5f;
    float const wGrid = (static_cast<float>(wIdx) + 0.5f) * sideOverW - 0.5f;

    // floor() (not truncation) so negative coordinates round the correct way.
    int64_t const hFloor = static_cast<int64_t>(floorf(hGrid));
    int64_t const wFloor = static_cast<int64_t>(floorf(wGrid));
    int64_t const hCeil = hFloor + 1;
    int64_t const wCeil = wFloor + 1;
    float const hFrac = hGrid - static_cast<float>(hFloor);
    float const wFrac = wGrid - static_cast<float>(wFloor);

    // Validity is computed on the UNclamped floor/ceil (the padding="zeros" mask).
    bool const hFloorValid = (hFloor >= 0) && (hFloor <= numGridPerSide - 1);
    bool const hCeilValid = (hCeil >= 0) && (hCeil <= numGridPerSide - 1);
    bool const wFloorValid = (wFloor >= 0) && (wFloor <= numGridPerSide - 1);
    bool const wCeilValid = (wCeil >= 0) && (wCeil <= numGridPerSide - 1);

    // Indices are gathered on the clamped taps.
    int64_t const hFloorC = std::min(std::max(hFloor, int64_t{0}), numGridPerSide - 1);
    int64_t const hCeilC = std::min(std::max(hCeil, int64_t{0}), numGridPerSide - 1);
    int64_t const wFloorC = std::min(std::max(wFloor, int64_t{0}), numGridPerSide - 1);
    int64_t const wCeilC = std::min(std::max(wCeil, int64_t{0}), numGridPerSide - 1);

    int64_t const baseH = hFloorC * numGridPerSide;
    int64_t const baseHCeil = hCeilC * numGridPerSide;
    int64_t const targetIdx = startIdx + tid;

    fastPosEmbedIdx[0 * totalSeqLength + targetIdx] = baseH + wFloorC;
    fastPosEmbedIdx[1 * totalSeqLength + targetIdx] = baseH + wCeilC;
    fastPosEmbedIdx[2 * totalSeqLength + targetIdx] = baseHCeil + wFloorC;
    fastPosEmbedIdx[3 * totalSeqLength + targetIdx] = baseHCeil + wCeilC;

    float const w0 = (1.0f - hFrac) * (1.0f - wFrac) * ((hFloorValid && wFloorValid) ? 1.0f : 0.0f);
    float const w1 = (1.0f - hFrac) * wFrac * ((hFloorValid && wCeilValid) ? 1.0f : 0.0f);
    float const w2 = hFrac * (1.0f - wFrac) * ((hCeilValid && wFloorValid) ? 1.0f : 0.0f);
    float const w3 = hFrac * wFrac * ((hCeilValid && wCeilValid) ? 1.0f : 0.0f);
    fastPosEmbedWeight[0 * totalSeqLength + targetIdx] = __float2half(w0);
    fastPosEmbedWeight[1 * totalSeqLength + targetIdx] = __float2half(w1);
    fastPosEmbedWeight[2 * totalSeqLength + targetIdx] = __float2half(w2);
    fastPosEmbedWeight[3 * totalSeqLength + targetIdx] = __float2half(w3);
}

void initFastPosEmbedMuseGlimmerViT(rt::Tensor& fastPosEmbedIdx, rt::Tensor& fastPosEmbedWeight,
    std::vector<int64_t> const& gridTHW, int64_t const numGridPerSide, int64_t const startIdx, cudaStream_t stream)
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
    int64_t const T = gridTHW[0];
    int64_t const H = gridTHW[1];
    int64_t const W = gridTHW[2];
    check::check(numGridPerSide > 0, "numGridPerSide must be positive.");
    float const sideOverH = static_cast<float>(numGridPerSide) / static_cast<float>(H);
    float const sideOverW = static_cast<float>(numGridPerSide) / static_cast<float>(W);

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (H * W + blockSize - 1) / blockSize;

    // Spatial interpolation repeats per temporal frame (raster order, offset by the frame's patch base).
    for (int64_t t = 0; t < T; ++t)
    {
        initFastPosEmbedMuseGlimmerKernel<<<gridSize, blockSize, 0, stream>>>(fastPosEmbedIdx.dataPointer<int64_t>(),
            fastPosEmbedWeight.dataPointer<half>(), H, W, numGridPerSide, sideOverH, sideOverW, startIdx + t * H * W,
            totalSeqLength);
    }
}

__global__ void transposeToPatchNemotronKernel(half const* blockPixels, half* inputPatches, int64_t const T,
    int64_t const C, int64_t const H, int64_t const W, int64_t const P, int64_t const totalElements)
{
    // Row layout: [group * numPatches + (pi * (W/P) + pj), ((t*C + c)*P + py)*P + px]
    // Source layout: [group*T + t, c, pi*P + py, pj*P + px]
    auto const tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= totalElements)
    {
        return;
    }

    int64_t const rowWidth = T * C * P * P;
    int64_t const gridW = W / P;
    int64_t const numPatches = (H / P) * gridW;

    int64_t const row = tid / rowWidth;
    int64_t const col = tid % rowWidth;

    int64_t const group = row / numPatches;
    int64_t const patch = row % numPatches;
    int64_t const pi = patch / gridW;
    int64_t const pj = patch % gridW;

    int64_t const t = col / (C * P * P);
    int64_t const c = (col % (C * P * P)) / (P * P);
    int64_t const py = (col % (P * P)) / P;
    int64_t const px = col % P;

    int64_t const srcIdx = (((group * T + t) * C + c) * H + pi * P + py) * W + pj * P + px;
    inputPatches[tid] = blockPixels[srcIdx];
}

void transposeToPatchNemotronViT(rt::Tensor const& blockPixels, rt::Tensor& inputPatches,
    int64_t const temporalPatchSize, int64_t const patchSize, cudaStream_t stream)
{
    check::check(
        blockPixels.getDeviceType() == rt::DeviceType::kGPU && inputPatches.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(blockPixels.getDataType() == DataType::kHALF && inputPatches.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(blockPixels.getShape().getNumDims() == 4, "blockPixels shape shall be [frames, channels, H, W].");
    check::check(temporalPatchSize > 0, "temporalPatchSize must be positive.");
    check::check(patchSize > 0, "patchSize must be positive.");

    int64_t const frames = blockPixels.getShape()[0];
    int64_t const channels = blockPixels.getShape()[1];
    int64_t const height = blockPixels.getShape()[2];
    int64_t const width = blockPixels.getShape()[3];
    check::check(frames % temporalPatchSize == 0, "Frame count must be a multiple of temporalPatchSize.");
    check::check(height % patchSize == 0 && width % patchSize == 0, "Image dims must be multiples of patchSize.");

    int64_t const totalElements = frames * channels * height * width;
    check::check(
        inputPatches.getShape().volume() >= totalElements, "inputPatches tensor too small for the patch output.");
    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    transposeToPatchNemotronKernel<<<gridSize, blockSize, 0, stream>>>(blockPixels.dataPointer<half>(),
        inputPatches.dataPointer<half>(), temporalPatchSize, channels, height, width, patchSize, totalElements);
}

__global__ void addPosEmbedNemotronKernel(
    half* patchEmbeds, half const* posEmbed, int64_t const perBlockElements, int64_t const totalElements)
{
    auto const tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= totalElements)
    {
        return;
    }
    patchEmbeds[tid] = __hadd(patchEmbeds[tid], posEmbed[tid % perBlockElements]);
}

void addPosEmbedNemotronViT(rt::Tensor& patchEmbeds, rt::Tensor const& posEmbed, cudaStream_t stream)
{
    check::check(
        patchEmbeds.getDeviceType() == rt::DeviceType::kGPU && posEmbed.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(patchEmbeds.getDataType() == DataType::kHALF && posEmbed.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(patchEmbeds.getShape().getNumDims() == 3 && posEmbed.getShape().getNumDims() == 2,
        "patchEmbeds shape shall be [numBlocks, numPatches, hidden] and posEmbed [numPatches, hidden].");
    check::check(
        patchEmbeds.getShape()[1] == posEmbed.getShape()[0] && patchEmbeds.getShape()[2] == posEmbed.getShape()[1],
        "posEmbed dims must match patchEmbeds per-block dims.");

    int64_t const perBlockElements = posEmbed.getShape().volume();
    check::check(perBlockElements > 0, "posEmbed must be non-empty.");
    int64_t const totalElements = patchEmbeds.getShape().volume();
    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    addPosEmbedNemotronKernel<<<gridSize, blockSize, 0, stream>>>(
        patchEmbeds.dataPointer<half>(), posEmbed.dataPointer<half>(), perBlockElements, totalElements);
}

__global__ void evsScoresNemotronKernel(
    half const* embeds, float* scores, int64_t const tokensPerGroup, int64_t const hidden)
{
    // One CTA per token (g, s). Cosine dissimilarity vs the same spatial slot in the previous
    // temporal group; group 0 gets the keep-always sentinel 255 (matches HF EVS).
    int64_t const token = blockIdx.x;
    int64_t const group = token / tokensPerGroup;

    if (group == 0)
    {
        if (threadIdx.x == 0)
        {
            scores[token] = 255.0F;
        }
        return;
    }

    half const* cur = embeds + token * hidden;
    half const* prev = embeds + (token - tokensPerGroup) * hidden;

    float dot = 0.0F;
    float normCur = 0.0F;
    float normPrev = 0.0F;
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x)
    {
        float const a = __half2float(cur[i]);
        float const b = __half2float(prev[i]);
        dot += a * b;
        normCur += a * a;
        normPrev += b * b;
    }

    __shared__ float sDot[32];
    __shared__ float sCur[32];
    __shared__ float sPrev[32];
    int const lane = threadIdx.x % 32;
    int const warp = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2)
    {
        dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
        normCur += __shfl_down_sync(0xFFFFFFFF, normCur, offset);
        normPrev += __shfl_down_sync(0xFFFFFFFF, normPrev, offset);
    }
    if (lane == 0)
    {
        sDot[warp] = dot;
        sCur[warp] = normCur;
        sPrev[warp] = normPrev;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float d = 0.0F;
        float nc = 0.0F;
        float np = 0.0F;
        int const numWarps = (blockDim.x + 31) / 32;
        for (int w = 0; w < numWarps; ++w)
        {
            d += sDot[w];
            nc += sCur[w];
            np += sPrev[w];
        }
        // torch cosine_similarity clamps each norm to eps before dividing.
        float const eps = 1e-8F;
        float const cos = d / (fmaxf(sqrtf(nc), eps) * fmaxf(sqrtf(np), eps));
        scores[token] = 1.0F - cos;
    }
}

void evsScoresNemotronViT(
    rt::Tensor const& embeds, rt::Tensor& scores, int64_t const tokensPerGroup, cudaStream_t stream)
{
    check::check(embeds.getDeviceType() == rt::DeviceType::kGPU && scores.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(embeds.getDataType() == DataType::kHALF && scores.getDataType() == DataType::kFLOAT,
        "Data type check failed for the input tensors.");
    check::check(embeds.getShape().getNumDims() == 2, "embeds shape shall be [numTokens, hidden].");
    check::check(tokensPerGroup > 0, "tokensPerGroup must be positive.");
    int64_t const numTokens = embeds.getShape()[0];
    check::check(numTokens % tokensPerGroup == 0, "numTokens must be a multiple of tokensPerGroup.");
    check::check(scores.getShape().volume() >= numTokens, "scores tensor too small.");

    int64_t const hidden = embeds.getShape()[1];
    evsScoresNemotronKernel<<<static_cast<uint32_t>(numTokens), 256, 0, stream>>>(
        embeds.dataPointer<half>(), scores.dataPointer<float>(), tokensPerGroup, hidden);
}

template <PixelDataType dtype>
struct AsPODType
{
};
template <>
struct AsPODType<PixelDataType::kHALF>
{
    typedef __half type;
};
enum class Parallel : unsigned int
{
    kSINGLE_PIXEL = 1
};

template <typename Scalar>
static __forceinline__ __device__ Scalar limit(Scalar value, Scalar low, Scalar high)
{
    return value < low ? low : (value > high ? high : value);
}

template <typename Scalar>
static __device__ __forceinline__ uint8_t u8cast(Scalar value)
{
    return value < 0 ? 0 : (value >= 255 ? 255 : uint8_t(value));
}

template <typename Scalar>
struct Saturate
{
};
template <>
struct Saturate<__half>
{
    __device__ __forceinline__ static __half cast(float x)
    {
        return __half(x);
    }
};

template <typename OutDType, Parallel parallel, PixelLayout layout>
struct DataLayoutInvoker
{
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NHWC RGB
template <typename OutDType>
struct DataLayoutInvoker<OutDType, Parallel::kSINGLE_PIXEL, PixelLayout::kNHWC_RGB>
{
    static __device__ __forceinline__ void call(
        OutDType* pdst, OutDType r, OutDType g, OutDType b, int ib, int x, int y, int stride, int height)
    {
        OutDType* p = pdst + (ib * height + y) * stride + x * 3;
        p[0] = r;
        p[1] = g;
        p[2] = b;
    }
};

// Two divisions in this order: folding them into one multiply by 1 / (255 * std) is algebraically
// equal but differs in the last bit.
template <typename Scalar>
static __device__ void __forceinline__ normalize_rgb(uint8_t r0, uint8_t g0, uint8_t b0, Scalar& r, Scalar& g,
    Scalar& b, float mean0, float mean1, float mean2, float std0, float std1, float std2)
{
    r = Saturate<Scalar>::cast((r0 / 255.0f - mean0) / std0);
    g = Saturate<Scalar>::cast((g0 / 255.0f - mean1) / std1);
    b = Saturate<Scalar>::cast((b0 / 255.0f - mean2) / std2);
}

static __device__ void __forceinline__ yuv2rgb(
    int y, int u, int v, YuvToRgbCoeffs const& coeffs, uint8_t& r, uint8_t& g, uint8_t& b)
{
    float const luma = coeffs.yScale * ((float) y - coeffs.yOffset);
    float const cb = (float) u - 128.0f;
    float const cr = (float) v - 128.0f;

    float const rf = luma + coeffs.crToR * cr;
    float const gf = luma + coeffs.cbToG * cb + coeffs.crToG * cr;
    float const bf = luma + coeffs.cbToB * cb;

    r = u8cast(limit(rf, 0.0f, 255.0f) + 0.5f);
    g = u8cast(limit(gf, 0.0f, 255.0f) + 0.5f);
    b = u8cast(limit(bf, 0.0f, 255.0f) + 0.5f);
}

// Taps are returned unconverted: the conversion runs once on the filtered result, not per tap.
template <SourceFormat format>
static __device__ uint8_t __forceinline__ load_luma_sample(void const* luma, int x, int y, int stride);

template <>
__device__ uint8_t __forceinline__ load_luma_sample<SourceFormat::kNV12PL>(void const* luma, int x, int y, int stride)
{
    return *((unsigned char const*) luma + (int64_t) y * stride + x);
}

template <>
__device__ uint8_t __forceinline__ load_luma_sample<SourceFormat::kNV12BL>(void const* luma, int x, int y, int stride)
{
    return tex2D<uint8_t>((cudaTextureObject_t) luma, x, y);
}

// Indexed on the chroma grid: x and y are chroma sample coordinates, not luma ones. The chroma plane
// carries its own row stride; sharing the luma stride reads the wrong rows on a padded frame.
template <SourceFormat format>
static __device__ uchar2 __forceinline__ load_chroma_sample(void const* chroma, int x, int y, int stride);

template <>
__device__ uchar2 __forceinline__ load_chroma_sample<SourceFormat::kNV12PL>(
    void const* chroma, int x, int y, int stride)
{
    unsigned char const* p = (unsigned char const*) chroma + (int64_t) y * stride + (int64_t) x * 2;
    return make_uchar2(p[0], p[1]);
}

template <>
__device__ uchar2 __forceinline__ load_chroma_sample<SourceFormat::kNV12BL>(
    void const* chroma, int x, int y, int stride)
{
    return tex2D<uchar2>((cudaTextureObject_t) chroma, x, y);
}

template <SourceFormat format>
static __device__ uchar3 __forceinline__ load_rgb_sample(void const* pixels, int x, int y, int stride);

template <>
__device__ uchar3 __forceinline__ load_rgb_sample<SourceFormat::kRGB8>(void const* pixels, int x, int y, int stride)
{
    unsigned char const* p = (unsigned char const*) pixels + (int64_t) y * stride + (int64_t) x * 3;
    return make_uchar3(p[0], p[1], p[2]);
}

// Source components per pixel, read by the host to size the shared memory stage and by the device to
// index it.
template <SourceFormat format>
constexpr int kPixelComponents = format == SourceFormat::kRGB8 ? 3 : 1;

// The tile is one warp wide; its height trades the halo rows its neighbour re-filters against the
// shared memory stage it needs.
constexpr int kTileWidth = 32;
constexpr int kTileHeight = 16;

// Shared memory for the staged rows. A vertical support taller than this is consumed in several
// chunks, so a steeper downscale costs more chunks rather than more shared memory.
constexpr int kTileStageBytes = 32 * 1024;

// Number of staged rows that fit the budget, at least one.
static inline int stage_rows_per_chunk(int const componentsPerRow)
{
    int const rows = kTileStageBytes / (int) (componentsPerRow * sizeof(float));
    return rows > 1 ? rows : 1;
}

// Rows the widest vertical support over a tile of this height reaches, used to size the stage.
static inline int tile_row_span(int const tileHeight, float const scale)
{
    float const fscale = scale > 1.0f ? scale : 1.0f;
    return (int) ceilf((tileHeight - 1) * scale + 4.0f * fscale) + 3;
}

// Catmull-Rom cubic. When downscaling, the support widens by the scale factor so the filter
// low-passes instead of applying a fixed four-tap kernel.
static __device__ float __forceinline__ catmull_rom_weight(float x)
{
    x = fabsf(x);
    if (x < 1.0f)
        return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    if (x < 2.0f)
        return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
    return 0.0f;
}

// One axis of the separable support: the source indices [first, last] that contribute to an output
// sample, the centre they are weighted around, and the widening factor.
struct SampleSupport
{
    int first;
    int last;
    float center;
    float fscale;
};

static __device__ SampleSupport __forceinline__ catmull_rom_support(int o, float scale, float shift)
{
    SampleSupport s;
    s.fscale = scale > 1.0f ? scale : 1.0f;
    s.center = (o + 0.5f) * scale + shift;
    s.first = (int) ceilf(s.center - 2.0f * s.fscale - 0.5f);
    s.last = (int) floorf(s.center + 2.0f * s.fscale - 0.5f);
    return s;
}

static __device__ uint8_t __forceinline__ quantize_u8(float v)
{
    return u8cast(limit(v, 0.0f, 255.0f) + 0.5f);
}

// Source rows [first, last] that output rows [y0, y0 + tileHeight) reach through the vertical support.
static __device__ void __forceinline__ tile_source_rows(
    int y0, int tileHeight, float scale, float fscale, int& first, int& last)
{
    first = (int) ceilf((y0 + 0.5f) * scale - 2.0f * fscale - 0.5f);
    last = (int) floorf((y0 + tileHeight - 1 + 0.5f) * scale + 2.0f * fscale - 0.5f);
}

// Horizontal half of the separable filter for one chunk of source rows, cooperatively over the block.
// Row r of the chunk holds source row chunkFirst + r filtered along x, normalised by its own weight
// sum; a row outside the frame carries the clamped edge row, which is what the vertical half expects.
template <SourceFormat format, int components>
static __device__ void __forceinline__ stage_rows(void const* plane, float* stage, int chunkFirst, int chunkRows,
    int x0, int cols, int outWidth, float scale, float shift, int srcWidth, int srcHeight, int stride, int batchOffset,
    int tid, int nthreads)
{
    for (int i = tid; i < chunkRows * cols; i += nthreads)
    {
        int const row = i / cols;
        int const col = i % cols;
        int const o = x0 + col;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        if (o < outWidth)
        {
            SampleSupport const h = catmull_rom_support(o, scale, shift);
            int const sy = limit(chunkFirst + row, 0, srcHeight - 1) + batchOffset;
            float wsum = 0.0f;
            for (int ix = h.first; ix <= h.last; ++ix)
            {
                float const w = catmull_rom_weight((ix + 0.5f - h.center) / h.fscale);
                int const sx = limit(ix, 0, srcWidth - 1);
                if constexpr (components == 3)
                {
                    uchar3 const s = load_rgb_sample<format>(plane, sx, sy, stride);
                    acc0 += w * (float) s.x;
                    acc1 += w * (float) s.y;
                    acc2 += w * (float) s.z;
                }
                else if constexpr (components == 2)
                {
                    uchar2 const s = load_chroma_sample<format>(plane, sx, sy, stride);
                    acc0 += w * (float) s.x;
                    acc1 += w * (float) s.y;
                }
                else
                {
                    acc0 += w * (float) load_luma_sample<format>(plane, sx, sy, stride);
                }
                wsum += w;
            }
            float const inv = wsum > 0.0f ? 1.0f / wsum : 0.0f;
            acc0 *= inv;
            acc1 *= inv;
            acc2 *= inv;
        }
        stage[i * components] = acc0;
        if constexpr (components >= 2)
            stage[i * components + 1] = acc1;
        if constexpr (components == 3)
            stage[i * components + 2] = acc2;
    }
}

// Vertical half over one staged chunk: adds the taps of this thread's support that the chunk holds.
template <int components>
static __device__ void __forceinline__ accumulate_column(float const* stage, SampleSupport const& v, int chunkFirst,
    int chunkRows, int cols, int col, float& acc0, float& acc1, float& acc2, float& wsum)
{
    int const from = v.first > chunkFirst ? v.first : chunkFirst;
    int const to = v.last < chunkFirst + chunkRows - 1 ? v.last : chunkFirst + chunkRows - 1;
    for (int iy = from; iy <= to; ++iy)
    {
        float const w = catmull_rom_weight((iy + 0.5f - v.center) / v.fscale);
        int const o = ((iy - chunkFirst) * cols + col) * components;
        acc0 += w * stage[o];
        if constexpr (components >= 2)
            acc1 += w * stage[o + 1];
        if constexpr (components == 3)
            acc2 += w * stage[o + 2];
        wsum += w;
    }
}

// What a YUV source has and a packed source has no equivalent of: the chroma plane, the addressing
// the luma extent and a single stride cannot describe, its chunk height, and the conversion matrix.
// Scales are against the output chroma grid ((out + 1) / 2), which is what the sampler indexes.
struct YuvSource
{
    void const* chroma;
    int chromaStride;
    int chromaWidth;
    int chromaHeight;
    float chromaScaleX;
    float chromaScaleY;
    float chromaShift; // horizontal sampling phase, in chroma samples
    int chromaChunkRows;
    YuvToRgbCoeffs coeffs;
};

struct NoYuvSource
{
};

template <SourceFormat format>
struct YuvSourceFor
{
    typedef YuvSource type;
};
template <>
struct YuvSourceFor<SourceFormat::kRGB8>
{
    typedef NoYuvSource type;
};

// Resize, convert and normalise one batch in a single launch, filtering separably through a shared
// memory stage: the block filters the source rows its output tile reaches along x, then each thread
// filters its own column along y. Quantising once after both halves keeps the [0, 255] saturation off
// the individual taps.
//
// A YUV source stages its two planes in turn and they share the buffer, so its size follows the wider.
template <SourceFormat source_format, typename OutDType, PixelLayout layout>
static __global__ void preprocess_image_kernel_1x(void const* src_plane, OutDType* pdst, float sx, float sy,
    int src_height, int src_width, int src_stride, typename YuvSourceFor<source_format>::type yuv, float mean0,
    float mean1, float mean2, float scale0, float scale1, float scale2, int dst_width, int dst_stride, int dst_height,
    int nbatch, int chunk_rows)
{
    extern __shared__ float stage[];

    constexpr bool is_rgb = source_format == SourceFormat::kRGB8;
    constexpr int pixel_components = kPixelComponents<source_format>;

    int const tile_height = blockDim.y;
    int const x0 = blockIdx.x * kTileWidth;
    int const y0 = blockIdx.y * tile_height;
    int const x = x0 + threadIdx.x;
    int const y = y0 + threadIdx.y;
    bool const active = x < dst_width && y < dst_height;

    int const tid = threadIdx.y * kTileWidth + threadIdx.x;
    int const nthreads = kTileWidth * tile_height;
    float const fsy = sy > 1.0f ? sy : 1.0f;

    SampleSupport const v = catmull_rom_support(y, sy, 0.0f);
    int srcFirst = 0;
    int srcLast = 0;
    tile_source_rows(y0, tile_height, sy, fsy, srcFirst, srcLast);

    for (int ib = blockIdx.z; ib < nbatch; ib += gridDim.z)
    {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float wsum = 0.0f;
        for (int chunk = srcFirst; chunk <= srcLast; chunk += chunk_rows)
        {
            int const rows = min(chunk_rows, srcLast - chunk + 1);
            __syncthreads();
            stage_rows<source_format, pixel_components>(src_plane, stage, chunk, rows, x0, kTileWidth, dst_width, sx,
                0.0f, src_width, src_height, src_stride, ib * src_height, tid, nthreads);
            __syncthreads();
            if (active)
                accumulate_column<pixel_components>(
                    stage, v, chunk, rows, kTileWidth, threadIdx.x, acc0, acc1, acc2, wsum);
        }
        float const inv = wsum > 0.0f ? 1.0f / wsum : 0.0f;

        uint8_t r0 = 0;
        uint8_t g0 = 0;
        uint8_t b0 = 0;
        if constexpr (is_rgb)
        {
            r0 = quantize_u8(acc0 * inv);
            g0 = quantize_u8(acc1 * inv);
            b0 = quantize_u8(acc2 * inv);
        }
        else
        {
            // Chroma is filtered on the output chroma grid, so the 2x2 luma block that shared a chroma
            // sample before the resize still shares one after it. x0 is a multiple of kTileWidth, so the
            // tile's output columns cover exactly kTileWidth / 2 output chroma columns.
            int const chroma_cols = kTileWidth / 2;
            float const cfsy = yuv.chromaScaleY > 1.0f ? yuv.chromaScaleY : 1.0f;
            int const co0 = y0 >> 1;
            int const coLast = (y0 + tile_height - 1) >> 1;
            int chromaFirst = 0;
            int chromaLast = 0;
            tile_source_rows(co0, coLast - co0 + 1, yuv.chromaScaleY, cfsy, chromaFirst, chromaLast);
            SampleSupport const cv = catmull_rom_support(y >> 1, yuv.chromaScaleY, 0.0f);

            float cb = 0.0f;
            float cr = 0.0f;
            float unused = 0.0f;
            float cwsum = 0.0f;
            for (int chunk = chromaFirst; chunk <= chromaLast; chunk += yuv.chromaChunkRows)
            {
                int const rows = min(yuv.chromaChunkRows, chromaLast - chunk + 1);
                __syncthreads();
                stage_rows<source_format, 2>(yuv.chroma, stage, chunk, rows, x0 >> 1, chroma_cols, (dst_width + 1) / 2,
                    yuv.chromaScaleX, yuv.chromaShift, yuv.chromaWidth, yuv.chromaHeight, yuv.chromaStride,
                    ib * yuv.chromaHeight, tid, nthreads);
                __syncthreads();
                if (active)
                    accumulate_column<2>(
                        stage, cv, chunk, rows, chroma_cols, (x >> 1) - (x0 >> 1), cb, cr, unused, cwsum);
            }
            float const cinv = cwsum > 0.0f ? 1.0f / cwsum : 0.0f;
            yuv2rgb(quantize_u8(acc0 * inv), quantize_u8(cb * cinv), quantize_u8(cr * cinv), yuv.coeffs, r0, g0, b0);
        }

        if (active)
        {
            OutDType r, g, b;
            normalize_rgb(r0, g0, b0, r, g, b, mean0, mean1, mean2, scale0, scale1, scale2);
            DataLayoutInvoker<OutDType, Parallel::kSINGLE_PIXEL, layout>::call(
                pdst, r, g, b, ib, x, y, dst_stride, dst_height);
        }
    }
}

template <SourceFormat source_format, PixelDataType out_dtype, PixelLayout layout>
void batched_preprocess_image_impl(void const* plane0, void const* plane1, int input_width, int stride0,
    int input_height, int input_batch, int stride1, float chroma_shift, YuvToRgbCoeffs coeffs, void* out_ptr,
    int out_width, int out_stride, int out_height, float mean0, float mean1, float mean2, float scale0, float scale1,
    float scale2, cudaStream_t stream)
{
    float sx = input_width / (float) out_width;
    float sy = input_height / (float) out_height;

    using OutDType = typename AsPODType<out_dtype>::type;
    constexpr int pixel_components = kPixelComponents<source_format>;

    // A chunk holds as many staged rows as the budget allows, capped at the rows the tile reaches.
    int const chunk_rows = min(tile_row_span(kTileHeight, sy), stage_rows_per_chunk(kTileWidth * pixel_components));
    int stage_floats = chunk_rows * kTileWidth * pixel_components;

    typename YuvSourceFor<source_format>::type yuv;
    if constexpr (source_format != SourceFormat::kRGB8)
    {
        int const chroma_cols = kTileWidth / 2;
        yuv.chroma = plane1;
        yuv.chromaStride = stride1;
        yuv.chromaWidth = (input_width + 1) / 2;
        yuv.chromaHeight = (input_height + 1) / 2;
        yuv.chromaScaleX = yuv.chromaWidth / (float) ((out_width + 1) / 2);
        yuv.chromaScaleY = yuv.chromaHeight / (float) ((out_height + 1) / 2);
        yuv.chromaShift = chroma_shift;
        yuv.chromaChunkRows
            = min(tile_row_span((kTileHeight + 1) / 2, yuv.chromaScaleY), stage_rows_per_chunk(chroma_cols * 2));
        yuv.coeffs = coeffs;
        stage_floats = max(stage_floats, yuv.chromaChunkRows * chroma_cols * 2);
    }

    int grid_z = input_batch >= 32 ? 32 : input_batch;
    dim3 dim_block(kTileWidth, kTileHeight);
    dim3 dim_grid((out_width + kTileWidth - 1) / kTileWidth, (out_height + kTileHeight - 1) / kTileHeight, grid_z);
    preprocess_image_kernel_1x<source_format, OutDType, layout>
        <<<dim_grid, dim_block, stage_floats * sizeof(float), stream>>>(plane0, (OutDType*) out_ptr, sx, sy,
            input_height, input_width, stride0, yuv, mean0, mean1, mean2, scale0, scale1, scale2, out_width, out_stride,
            out_height, input_batch, chunk_rows);
    CUDA_CHECK(cudaPeekAtLastError());
}
typedef void (*batched_preprocess_image_impl_function)(void const* plane0, void const* plane1, int input_width,
    int stride0, int input_height, int input_batch, int stride1, float chroma_shift, YuvToRgbCoeffs coeffs,
    void* out_ptr, int out_width, int out_stride, int out_height, float mean0, float mean1, float mean2, float scale0,
    float scale1, float scale2, cudaStream_t stream);

static_assert(static_cast<int>(SourceFormat::kNV12BL) == 1 && static_cast<int>(SourceFormat::kNV12PL) == 2
        && static_cast<int>(SourceFormat::kRGB8) == 3,
    "func_list is indexed by SourceFormat - 1; DefineSourceFormat lists the formats in that order.");

#define DefineSourceFormat(...)                                                                                        \
    batched_preprocess_image_impl<SourceFormat::kNV12BL, __VA_ARGS__>,                                                 \
        batched_preprocess_image_impl<SourceFormat::kNV12PL, __VA_ARGS__>,                                             \
        batched_preprocess_image_impl<SourceFormat::kRGB8, __VA_ARGS__>,

#define DefineDType(...) DefineSourceFormat(PixelDataType::kHALF, __VA_ARGS__)

#define DefineLayout DefineDType(PixelLayout::kNHWC_RGB)

#define DefineAllFunction DefineLayout

template <typename T>
struct EnumCount
{
};
template <>
struct EnumCount<SourceFormat>
{
    static int const value = 3;
};
template <>
struct EnumCount<PixelDataType>
{
    static int const value = 1;
};
template <>
struct EnumCount<PixelLayout>
{
    static int const value = 1;
};
template <>
struct EnumCount<Interpolation>
{
    static int const value = 1;
};

static batched_preprocess_image_impl_function const func_list[] = {DefineAllFunction nullptr};

static_assert(sizeof(func_list) / sizeof(func_list[0]) - 1
        == EnumCount<Interpolation>::value * EnumCount<PixelLayout>::value * EnumCount<PixelDataType>::value
            * EnumCount<SourceFormat>::value,
    "func_list holds one instantiation per (interpolation, layout, dtype, source format) combination.");

void batchedPreprocessImage(void const* const plane0, void const* const plane1, int const input_width,
    int const stride0, int const input_height, int const input_batch, SourceFormat const source_format,
    int const stride1, float const chroma_shift, YuvToRgbCoeffs const coeffs, void* const out_ptr, int const out_width,
    int const out_stride, int const out_height, PixelDataType const out_dtype, PixelLayout const out_layout,
    Interpolation const interp, float const mean0, float const mean1, float const mean2, float const scale0,
    float const scale1, float const scale2, cudaStream_t stream)
{
    int const iformat = (int) source_format - 1;
    int const odtype = (int) out_dtype - 1;
    int const olayout = (int) out_layout - 1;
    int const iinterp = (int) interp - 1;
    int const index = ((iinterp * EnumCount<PixelLayout>::value + olayout) * EnumCount<PixelDataType>::value + odtype)
            * EnumCount<SourceFormat>::value
        + iformat;
    int const instantiated = (int) (sizeof(func_list) / sizeof(func_list[0])) - 1;
    if (iformat < 0 || iformat >= EnumCount<SourceFormat>::value || odtype < 0
        || odtype >= EnumCount<PixelDataType>::value || olayout < 0 || olayout >= EnumCount<PixelLayout>::value
        || iinterp < 0 || iinterp >= EnumCount<Interpolation>::value || index < 0 || index >= instantiated)
    {
        ELLM_CHECK(
            false, "batchedPreprocessImage: no kernel instantiated for dispatch index " + std::to_string(index) + ".");
    }

    batched_preprocess_image_impl_function func = func_list[index];
    func(plane0, plane1, input_width, stride0, input_height, input_batch, stride1, chroma_shift, coeffs, out_ptr,
        out_width, out_stride, out_height, mean0, mean1, mean2, scale0, scale1, scale2, stream);
}

} // namespace kernel
} // namespace trt_edgellm
