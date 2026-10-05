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

#pragma once

#include "common/tensor.h"
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{
//! The kernel will transpose image data to patch format for Gemma4 VIT (channel-last within patch)
//! Gemma4's vision encoder expects patches with element order [patchH, patchW, C] matching
//! HuggingFace convert_image_to_patches: reshape(C,pH,ps,pW,ps).permute(1,3,2,4,0).reshape(pH*pW,-1)
//! Inputs:
//!     originalImage [GPU, Half]: Current image [1, height, width, channels]
//!     inputOffset: Offset in elements into inputPatches (prevCuSeqlen * inputDim)
//!     patchSize: Patch size for the vision transformer
//!     stream: CUDA stream for execution
//! Outputs:
//!     inputPatches [GPU, Half]: Total VIT input tensor [totalSeqLength, inputDim]
//!         inputDim = patchSize * patchSize * channels
//! \throws std::runtime_error if image has invalid shape, data type or location
void transposeToPatchGemma4ViT(rt::Tensor const& originalImage, rt::Tensor& inputPatches, int64_t const inputOffset,
    int64_t const patchSize, cudaStream_t stream);

//! The kernel will transpose image data to patch format for Qwen2-VL and Qwen2.5-VL VIT
//! The transpose is corresponding to the following python code:
//! https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L299
//! Inputs:
//!     originalImage [GPU, Half]: Current image [T, height, width, channels]
//!     inputOffset: Offset of the input patches, denoting the start index of the current image
//!     temporalPatchSize: Temporal patch size for the vision transformer
//!     patchSize: Patch size for the vision transformer
//!     mergeSize: Merge size for the vision transformer
//!     temporalFirst: Flatten patches as [T,C,H,W] instead of [C,T,H,W]
//!     channelLast: Flatten patches as [H,W,C,T]; mutually exclusive with temporalFirst
//!     stream: CUDA stream for execution
//! Outputs:
//!     inputPatches [GPU, Half]: Total VIT input tensor of all images [totalSeqLength, inputDim]
//!         curSeqLength = gridT * gridH * gridW * mergeSize * mergeSize
//!         totalSeqLength = sum(curSeqLength) over all images
//!         inputDim = channels * temporalPatchSize * patchSize * patchSize
//! \throws std::runtime_error if image has invalid shape, data type or location
void transposeToPatchQwenViT(rt::Tensor const& originalImage, rt::Tensor& inputPatches, int64_t const inputOffset,
    int64_t const temporalPatchSize, int64_t const patchSize, int64_t const mergeSize, bool temporalFirst,
    bool channelLast, cudaStream_t stream);

//! The kernel will initialize the rotary position embeddings for Qwen2.5-VL VIT
//! Inputs:
//!     gridTHW: Image grid dimensions [T, H, W] (Temporal, Height, Width)
//!     mergeSize: Merge size for the vision transformer
//!     startIdx: Start index for the current image
//!     rotaryBaseFrequency: Rotary base frequency
//!     scale: Scale for the rotary position embeddings
//!     stream: CUDA stream for execution
//! Outputs:
//!     rotaryPosEmb [GPU, Float]: Rotary position embeddings tensor [totalSeqLength, vitPosEmbDim]
//! \throws std::runtime_error if image has invalid shape, data type or location
void initRotaryPosEmbQwenViT(rt::Tensor& rotaryPosEmb, std::vector<int64_t> const& gridTHW, int64_t const mergeSize,
    int64_t const startIdx, float const rotaryBaseFrequency, float const scale, cudaStream_t stream);

//! The kernel will initialize the rotary position embeddings for Muse-Glimmer VIT.
//! Unlike initRotaryPosEmbQwenViT (which lays out per-token frequencies as concat(freq_h, freq_w) in
//! 2x2-merge-grouped token order with no position offset), Muse-Glimmer runs its encoder at
//! spatial_merge_size == 1 (raster token order) and its reference RoPE lays each token's row out as
//! concat(freq_w, freq_h) with a +1 position offset (mirrors position_ids.flip(-1) + 1). The rotary
//! frequency itself (inv_freq = 1 / theta^(2*k / vitPosEmbDim)) is identical; only the layout differs.
//! Inputs:
//!     gridTHW: Image grid dimensions [T, H, W] in patch units (H, W are the full patch grid, not merged)
//!     startIdx: Start patch index for the current image (raster token offset)
//!     rotaryBaseFrequency: Rotary base frequency (theta)
//!     stream: CUDA stream for execution
//! Outputs:
//!     rotaryPosEmb [GPU, Float]: Rotary position embeddings tensor [totalSeqLength, vitPosEmbDim];
//!         each row = concat(freq_w[vitPosEmbDim/2], freq_h[vitPosEmbDim/2]).
//! \throws std::runtime_error if the tensor has an invalid shape, data type or location
void initRotaryPosEmbMuseGlimmerViT(rt::Tensor& rotaryPosEmb, std::vector<int64_t> const& gridTHW,
    int64_t const startIdx, float const rotaryBaseFrequency, cudaStream_t stream);

//! The kernel will initialize the fast (interpolated) position embeddings for Muse-Glimmer VIT.
//! Unlike initFastPosEmbedQwenViT (align_corners == True mapping over 2x2-merge-grouped tokens), Muse-Glimmer
//! interpolates its num_grid_per_side x num_grid_per_side learned position table with align_corners == False
//! and "zeros" padding over raster (spatial_merge_size == 1) tokens, matching the reference
//! get_vision_bilinear_indices_and_weights / F.grid_sample(align_corners=False, padding="zeros").
//! Out-of-range floor/ceil taps are index-clamped but their weights are zeroed (the padding="zeros" behavior).
//! Inputs:
//!     gridTHW: Image grid dimensions [T, H, W] in patch units (only H and W drive the interpolation)
//!     numGridPerSide: Side length of the square learned position table (pos_emb_height == pos_emb_width)
//!     startIdx: Start patch index for the current image (raster token offset)
//!     stream: CUDA stream for execution
//! Outputs:
//!     fastPosEmbedIdx [GPU, Int64]: Fast position embeddings index tensor [4, totalSeqLength]
//!     fastPosEmbedWeight [GPU, Half]: Fast position embeddings weight tensor [4, totalSeqLength]
//! \throws std::runtime_error if a tensor has an invalid shape, data type or location
void initFastPosEmbedMuseGlimmerViT(rt::Tensor& fastPosEmbedIdx, rt::Tensor& fastPosEmbedWeight,
    std::vector<int64_t> const& gridTHW, int64_t const numGridPerSide, int64_t const startIdx, cudaStream_t stream);

//! The kernel will initialize Gemma4 vision 2-D rotary angle embeddings from pixel position ids
//! Inputs:
//!     pixelPositionIds [GPU, Int64]: Pixel position ids [totalSeqLength, 2] (x, y)
//!     rotaryBaseFrequency: Rotary base frequency
//!     stream: CUDA stream for execution
//! Outputs:
//!     rotaryPosEmb [GPU, Float]: Rotary angle embeddings tensor [totalSeqLength, headDim]
//! \throws std::runtime_error if tensors have invalid shape, data type or location
void initRotaryPosEmbGemma4ViT(
    rt::Tensor& rotaryPosEmb, rt::Tensor const& pixelPositionIds, float rotaryBaseFrequency, cudaStream_t stream);

//! The kernel will initialize Gemma4 vision dense pooling weights on GPU
//! Inputs:
//!     patchStart: First patch column for the current image in the packed patch tensor
//!     softStart: First soft-token row for the current image in the pooled output
//!     patchHeight: Patch-grid height for the current image
//!     patchWidth: Patch-grid width for the current image
//!     poolingKernelSize: Spatial pooling kernel size
//!     stream: CUDA stream for execution
//! Outputs:
//!     poolingWeights [GPU, Half]: Dense pooling weight tensor [totalSoftTokens, totalPatches]
//! \throws std::runtime_error if tensors have invalid shape, data type or location
void initPoolingWeightsGemma4ViT(rt::Tensor& poolingWeights, int64_t patchStart, int64_t softStart, int64_t patchHeight,
    int64_t patchWidth, int64_t poolingKernelSize, cudaStream_t stream);

//! The kernel will transpose image data to patch format for InternVL VIT
//! Inputs:
//!     originalImage [GPU, Half]: Current image [1, height, width, channels]
//!     inputOffset: Offset of the input patches, denoting the start index of the current image
//!     stream: CUDA stream for execution
//! Outputs:
//!     inputPatches [GPU, Half]: Total VIT input tensor of all images [totalNumBlocks, channels, blockSizeH,
//!     blockSizeW]
//!         curNumBlocks = blockH * blockW
//!         totalNumBlocks = sum(curNumBlocks) over all images
//! \throws std::runtime_error if image has invalid shape, data type or location
void transposeToPatchInternVLPhi4MM(
    rt::Tensor const& originalImage, rt::Tensor& inputPatches, int64_t const inputOffset, cudaStream_t stream);

//! The kernel will initialize the fast position embeddings for Qwen3-VL VIT
//! Inputs
//!     gridTHW: Image grid dimensions [T, H, W] (only H and W are used)
//!     mergeSize: Merge size for the vision transformer
//!     numGridPerSide: Number of grid per side for the vision transformer
//!     startIdx: Start index for the image
//!     stream: CUDA stream for execution
//! Outputs:
//!     fastPosEmbedIdx [GPU, Int64]: Fast position embeddings index tensor [4, totalSeqLength]
//!     fastPosEmbedWeight [GPU, Half]: Fast position embeddings weight tensor [4, totalSeqLength]
//! \throws std::runtime_error if image has invalid shape, data type or location
void initFastPosEmbedQwenViT(rt::Tensor& fastPosEmbedIdx, rt::Tensor& fastPosEmbedWeight,
    std::vector<int64_t> const& gridTHW, int64_t const mergeSize, int64_t const numGridPerSide, int64_t const startIdx,
    cudaStream_t stream);

//! Initialize learned-position interpolation metadata for Cosmos3-Edge VIT.
//! Cosmos3 uses PyTorch bilinear interpolation with align_corners=false, unlike Qwen3-VL's
//! align_corners=true coordinate mapping. Tensor layout and merge-group ordering match
//! initFastPosEmbedQwenViT.
void initFastPosEmbedCosmos3ViT(rt::Tensor& fastPosEmbedIdx, rt::Tensor& fastPosEmbedWeight,
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

//! The kernel will transpose block-split CHW pixels to patch format for the Nemotron-Omni VIT
//! (the runtime patch-embedder GEMM input). Groups of T consecutive frames are packed into one
//! row set; per-patch element order is (t, c, py, px) C-major, matching RADIO Im2Patches on
//! T-channel-stacked frames (T == 1 for still image tiles).
//! Inputs:
//!     blockPixels [GPU, Half]: [numFrames, channels, height, width]
//!         numFrames must be a multiple of temporalPatchSize
//!     temporalPatchSize: Frames packed per patch row (T)
//!     patchSize: Patch size for the vision transformer (P)
//!     stream: CUDA stream for execution
//! Outputs:
//!     inputPatches [GPU, Half]: [numFrames/T * numPatches, T*channels*P*P]
//!         numPatches = (height/P) * (width/P)
//! \throws std::runtime_error if tensors have invalid shape, data type or location
void transposeToPatchNemotronViT(rt::Tensor const& blockPixels, rt::Tensor& inputPatches,
    int64_t const temporalPatchSize, int64_t const patchSize, cudaStream_t stream);

//! The kernel adds a per-grid position embedding to every block of patch
//! embeddings (broadcast over the block dimension).
//! Inputs:
//!     patchEmbeds [GPU, Half]: [numBlocks, numPatches, hidden] (modified in place)
//!     posEmbed [GPU, Half]: [numPatches, hidden]
//!     stream: CUDA stream for execution
//! \throws std::runtime_error if tensors have invalid shape, data type or location
void addPosEmbedNemotronViT(rt::Tensor& patchEmbeds, rt::Tensor const& posEmbed, cudaStream_t stream);

//! The kernel computes Efficient Video Sampling dissimilarity scores over the
//! projected video embeddings: score[g][s] = 1 - cos(embeds[g][s], embeds[g-1][s])
//! for temporal groups g > 0, and the keep-always sentinel 255 for g == 0.
//! (Row compaction by the retained indices reuses kernel::embeddingLookup.)
//! Inputs:
//!     embeds [GPU, Half]: [numGroups * tokensPerGroup, hidden]
//!     tokensPerGroup: Spatial token count per temporal group
//!     stream: CUDA stream for execution
//! Outputs:
//!     scores [GPU, Float]: [numGroups * tokensPerGroup]
//! \throws std::runtime_error if tensors have invalid shape, data type or location
void evsScoresNemotronViT(
    rt::Tensor const& embeds, rt::Tensor& scores, int64_t const tokensPerGroup, cudaStream_t stream);

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
//! \throws std::runtime_error invalid tensor shape, location or data type
void phi4mmPostprocessVisionTokens(rt::Tensor const& srcEmbedding, rt::Tensor& dstEmbedding, Phi4MMIndex const& indices,
    Phi4MMGN const& gn, int64_t totalOutTokens, cudaStream_t stream);

enum class PixelLayout : unsigned int
{
    kNONE = 0,
    kNHWC_RGB = 1
};

enum class PixelDataType : unsigned int
{
    kNONE = 0,
    kHALF = 1
};

enum class Interpolation : unsigned int
{
    kNONE = 0,
    kBICUBIC = 1 //!< Catmull-Rom cubic, anti-aliased on downscale.
};

enum class SourceFormat : unsigned int
{
    kNONE = 0,
    kNV12BL = 1,
    kNV12PL = 2,
    kRGB8 = 3
};

//! YCbCr to RGB coefficients in 8-bit code units, applied as given with the signs folded in:
//!     R = yScale * (Y - yOffset) + crToR * (Cr - 128)
//!     G = yScale * (Y - yOffset) + cbToG * (Cb - 128) + crToG * (Cr - 128)
//!     B = yScale * (Y - yOffset) + cbToB * (Cb - 128)
struct YuvToRgbCoeffs
{
    float yScale;
    float yOffset;
    float crToR;
    float cbToG;
    float crToG;
    float cbToB;
};

//! Resize, convert and normalise `input_batch` frames in a single pass, writing `out_dtype` in
//! `out_layout`. `plane0` is the luma plane for the NV12 formats and the packed RGB frame for kRGB8,
//! `plane1` the chroma plane; both are cudaTextureObject_t for kNV12BL and plane pointers otherwise.
//! `stride0` and `stride1` are row strides in bytes, `out_stride` in elements. `chroma_shift` is the
//! horizontal chroma siting phase in chroma samples; `coeffs` is ignored for kRGB8.
//!
//! Frames stack vertically and the taps are clamped to the frame, so the caller guarantees, in bytes
//! from each pointer (kNV12BL excepted, being sampled):
//!     plane0  (input_batch * input_height - 1) * stride0 + input_width * (source_format == kRGB8 ? 3 : 1)
//!     plane1  (input_batch * ((input_height + 1) / 2) - 1) * stride1 + 2 * ((input_width + 1) / 2)
//!     out_ptr (input_batch * out_height - 1) * out_stride + out_width * 3 elements
//!
//! \throws std::runtime_error if that combination of format, dtype, layout and interpolation is not
//!     instantiated.
void batchedPreprocessImage(void const* const plane0, void const* const plane1, int const input_width,
    int const stride0, int const input_height, int const input_batch, SourceFormat const source_format,
    int const stride1, float const chroma_shift, YuvToRgbCoeffs const coeffs, void* const out_ptr, int const out_width,
    int const out_stride, int const out_height, PixelDataType const out_dtype, PixelLayout const out_layout,
    Interpolation const interp, float const mean0, float const mean1, float const mean2, float const scale0,
    float const scale1, float const scale2, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
