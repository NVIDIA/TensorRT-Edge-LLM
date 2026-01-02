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

#pragma once

#include "common/tensor.h"
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{
// The kernel will normalize image data and convert to half
// Inputs:
//     originalImage [GPU, UInt8]: [batch, height, width, channels]
//     mean [GPU, Float]: [channels]
//     std [GPU, Float]: [channels]
//     stream: CUDA stream for execution
// Outputs:
//     normalizedImage [GPU, Half]: [batch, height, width, channels]
void normalizeImage(rt::Tensor const& originalImage, rt::Tensor const& mean, rt::Tensor const& std,
    rt::Tensor& normalizedImage, cudaStream_t stream);

// The kernel will transpose image data to patch format for Qwen2-VL and Qwen2.5-VL VIT
// The transpose is corresponding to the following python code:
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L299
// Inputs:
//     originalImage [GPU, Half]: Current image [T, height, width, channels]
//     inputOffset: Offset of the input patches, denoting the start index of the current image
//     temporalPatchSize: Temporal patch size for the vision transformer
//     patchSize: Patch size for the vision transformer
//     mergeSize: Merge size for the vision transformer
//     stream: CUDA stream for execution
// Outputs:
//     inputPatches [GPU, Half]: Total VIT input tensor of all images [totalSeqLength, inputDim]
//         curSeqLength = gridT * gridH * gridW * mergeSize * mergeSize
//         totalSeqLength = sum(curSeqLength) over all images
//         inputDim = channels * temporalPatchSize * patchSize * patchSize
void transposeToPatchQwenViT(rt::Tensor const& originalImage, rt::Tensor& inputPatches, int64_t const inputOffset,
    int64_t const temporalPatchSize, int64_t const patchSize, int64_t const mergeSize, cudaStream_t stream);

// The kernel will initialize the attention mask for Qwen2-VL and Qwen2.5-VL VIT
// Inputs:
//     cuSeqlens [GPU, Int64]: Cumulative sequence lengths [num]
//     stream: CUDA stream for execution
// Outputs:
//     attentionMask [GPU, Half]: Attention mask tensor [1, curHW, curHW]
void initAttentionMaskQwenViT(rt::Tensor const& cuSeqlens, rt::Tensor& attentionMask, cudaStream_t stream);

// The kernel will initialize the rotary position embeddings for Qwen2.5-VL VIT
// Inputs:
//     gridTHW: Image grid dimensions [T, H, W] (Temporal, Height, Width)
//     mergeSize: Merge size for the vision transformer
//     startIdx: Start index for the current image
//     rotaryBaseFrequency: Rotary base frequency
//     scale: Scale for the rotary position embeddings
//     stream: CUDA stream for execution
// Outputs:
//     rotaryPosEmb [GPU, Float]: Rotary position embeddings tensor [totalSeqLength, vitPosEmbDim]
void initRotaryPosEmbQwenViT(rt::Tensor& rotaryPosEmb, std::vector<int64_t> const& gridTHW, int64_t const mergeSize,
    int64_t const startIdx, float const rotaryBaseFrequency, float const scale, cudaStream_t stream);

// The kernel will transpose image data to patch format for InternVL VIT
// Inputs:
//     originalImage [GPU, Half]: Current image [1, height, width, channels]
//     inputOffset: Offset of the input patches, denoting the start index of the current image
//     stream: CUDA stream for execution
// Outputs:
//     inputPatches [GPU, Half]: Total VIT input tensor of all images [totalNumBlocks, channels, blockSizeH, blockSizeW]
//         curNumBlocks = blockH * blockW
//         totalNumBlocks = sum(curNumBlocks) over all images
void transposeToPatchInternVLPhi4MM(
    rt::Tensor const& originalImage, rt::Tensor& inputPatches, int64_t const inputOffset, cudaStream_t stream);

// The kernel will initialize the fast position embeddings for Qwen3-VL VIT
// Inputs
//     gridTHW: Image grid dimensions [T, H, W] (only H and W are used)
//     mergeSize: Merge size for the vision transformer
//     numGridPerSide: Number of grid per side for the vision transformer
//     startIdx: Start index for the image
//     stream: CUDA stream for execution
// Outputs:
//     fastPosEmbedIdx [GPU, Int64]: Fast position embeddings index tensor [4, totalSeqLength]
//     fastPosEmbedWeight [GPU, Half]: Fast position embeddings weight tensor [4, totalSeqLength]
void initFastPosEmbedQwenViT(rt::Tensor& fastPosEmbedIdx, rt::Tensor& fastPosEmbedWeight,
    std::vector<int64_t> const& gridTHW, int64_t const mergeSize, int64_t const numGridPerSide, int64_t const startIdx,
    cudaStream_t stream);

//! Phi4MMIndex
//! Device-side index and size metadata for Phi-4MM HD packing.
//! Fields:
//! - hBlocks/wBlocks [numImages]: per-image grid sizes (hb = H/blockImageSizeH, wb = W/blockImageSizeW)
//! - srcGlbStart    [numImages]: starting raw-token offset for the tokensPerSide x tokensPerSide global grid of image i
//! - srcSubStart    [numImages]: starting raw-token offset for sub-grid tokens of image i
//! - dstOutStart    [numImages]: starting packed-token offset in dst for image i
//! - subOutLen      [numImages]: sub segment token count per image (includes one newline per row)
//! - numImages: batch size
//! - hidden: embedding length
//! - totalOutTokens: total tokens to be written across all images
struct Phi4MMIndex
{
    int32_t const* hBlocks;     // [numImages]
    int32_t const* wBlocks;     // [numImages]
    int64_t const* srcGlbStart; // [numImages]
    int64_t const* srcSubStart; // [numImages]
    int64_t const* dstOutStart; // [numImages]
    int64_t const* subOutLen;   // [numImages]
    int32_t numImages;
    int32_t hidden;
    int64_t totalOutTokens;
};

//! Phi4MMGN
//! Grid Newline (GN) and separator embeddings.
//! - subGN [hidden] FP16: newline token vector inserted at the end of each sub-grid row
//! - glbGN [hidden] FP16: single separator token placed between sub and global segments
struct Phi4MMGN
{
    half const* subGN; // [hidden]
    half const* glbGN; // [hidden]
};

constexpr int64_t kTokensPerBlockPhi4 = 256;
constexpr int64_t kTokensPerSidePhi4 = 16;

//! phi4mmPostprocessVisionTokens
//! Purpose:
//!   Construct the Phi-4MM HD image token sequence for a batch by gathering
//!   from raw ViT tokens and inserting Grid Newline (GN) separators.
//!
//! Inputs:
//!   - src: [numViTTokens, hidden] FP16
//!       Raw ViT tokens for all images (global + sub), concatenated across images.
//!   - dst: [totalOutTokens, hidden] FP16
//!       Output buffer for the packed HD sequence.
//!   - idx: Phi4MMIndex (device indices and sizes)
//!   - gn:  Phi4MMGN (newline and separator embeddings)
//! Output layout per image (contiguous in `dst`):
//!   1) Sub segment: rows = tokensPerSide*hb, cols = tokensPerSide*wb, strideOut = cols+1; last col is subGN (newline).
//!      Non-newline positions gather from src via (srcSubStart + blockId*256 + patchId).
//!   2) One glb_GN token (glbGN).
//!   3) Global segment: 16x16 grid with strideOut = 17; last col is subGN; others gather from srcGlbStart.
//!
//! Launch config:
//!   - gridDim.x = idx.totalOutTokens, blockDim.x = 128
//!   - Each CUDA block writes one output token vector; threads cooperate to copy `idx.hidden` elements.
void phi4mmPostprocessVisionTokens(rt::Tensor const& srcEmbedding, rt::Tensor& dstEmbedding, Phi4MMIndex const& indices,
    Phi4MMGN const& gn, int64_t totalOutTokens, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
