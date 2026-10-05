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

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

/*!
 * @brief Source pixel format and memory layout of an image buffer.
 *
 * Describes the input side only; the pipeline always produces packed RGB8.
 */
enum class ImageFormat : int32_t
{
    kRGB8 = 0, //!< Single plane, packed [H, W, 3], pitch-linear.
    kNV12PL,   //!< Two pitch-linear planes: Y8 [H, W], then UV8 interleaved [(H+1)/2, (W+1)/2].
    kNV12BL,   //!< As kNV12PL but hardware block-linear (512 B GOB = 64 B x 8 rows).
};

//! @brief YUV-to-RGB matrix. Required for YUV formats, ignored for kRGB8. kUnspecified is the
//!        value-initialised state and is rejected for YUV formats.
enum class ColorStandard : int32_t
{
    kUnspecified = 0,
    kBt601,
    kBt709,
    kBt2020
};

//! @brief Luma and chroma excursion. kLimited puts Y in [16, 235], kFull in [0, 255]. kUnspecified is
//!        the value-initialised state and is rejected for YUV formats.
enum class ColorRange : int32_t
{
    kUnspecified = 0,
    kLimited,
    kFull
};

/*!
 * @brief How to read the bytes of ImageData::buffer: plane placement and alignment padding.
 *
 * Valid bytes per row, excluding the padding pitchBytes may add, for a frame of W x H:
 *   kRGB8  plane 0: H rows, 3W.  kNV12*  plane 0: H rows, W (Y);
 *                                        plane 1: (H+1)/2 rows, 2 * ((W+1)/2) (interleaved CbCr).
 * Frame t of plane p starts at rawPointer() + planeOffsetBytes[p] + t * frameStrideBytes[p]; all
 * planes and frames live in one allocation.
 */
struct ImageLayout
{
    ImageFormat format{ImageFormat::kRGB8};
    ColorStandard colorStandard{}; //!< Ignored for kRGB8. Required for YUV formats.
    ColorRange colorRange{};       //!< Ignored for kRGB8. Required for YUV formats.

    std::array<int64_t, 2> planeOffsetBytes{}; //!< Plane 1 is CbCr; unused for kRGB8.
    std::array<int64_t, 2> pitchBytes{};       //!< Row stride, including any alignment padding.
    std::array<int64_t, 2> frameStrideBytes{}; //!< Ignored when frames == 1.
};

/*!
 * @brief Texture objects over the planes of a block-linear frame the caller imported into CUDA.
 *
 * Block-linear pixels have no address a kernel can walk, so they are sampled rather than loaded. The
 * caller owns the objects and the arrays behind them. An unset plane is zero.
 */
struct ImageTexturePlanes
{
    cudaTextureObject_t plane0{}; //!< Luma: one uint8 sample per texel over width x height.
    cudaTextureObject_t plane1{}; //!< Chroma: an interleaved CbCr pair over ((w + 1) / 2) x ((h + 1) / 2).
};

/*!
 * @brief Image data container (image or video frame stack)
 *
 * Exactly one of `buffer` and `textures` carries the pixels: kNV12BL uses `textures`, every other
 * format uses `buffer`, read through `layout`. The decode entry points produce a uint8 RGB tensor
 * shaped `[frames, height, width, channels]`; wrapped caller-owned storage may instead hold NV12
 * planes in one flat allocation. `channels` is always 3.
 */
class ImageData
{
public:
    std::shared_ptr<rt::Tensor> buffer; //!< Pixel storage (UINT8); read it through `layout`. Null for kNV12BL.
    ImageTexturePlanes textures;        //!< Pixel storage for kNV12BL; unset for every other format.
    ImageLayout layout;                 //!< Format and colour metadata; for `buffer`, also its plane addressing.
    int64_t width{0};                   //!< Image width
    int64_t height{0};                  //!< Image height
    int64_t channels{0};                //!< Number of channels (e.g., 3 for RGB)
    int64_t frames{1};                  //!< Number of frames (T); the modality is flagged by isVideo.
    double fps{1.0};                    //!< Video sample fps for MRoPE timestamps; ignored unless isVideo.
    bool doResize{true};                //!< When false, the vision runner skips its internal resize.
    bool isVideo{false};                //!< Explicit modality: a single-frame video is still a video.
    std::vector<double> timestamps;     //!< Optional source timestamps (seconds); empty assumes uniform fps spacing.

    /*!
     * @brief Default constructor (creates uninitialized ImageData)
     */
    ImageData() noexcept = default;

    /*!
     * @brief Construct image data
     * @param data Image tensor with shape [T, H, W, C]. Single-frame still images use T=1.
     * @throws std::runtime_error if tensor content not UINT8, tensor shape not 4D, or number of channels not 3
     */
    ImageData(rt::Tensor&& data);

    //! @brief Base of the pixels, for consumers that read them as one packed host-RGB run.
    //! @return Pointer to that run, or nullptr unless `buffer` is host-resident kRGB8.
    unsigned char* data() const noexcept;

    //! @brief Reach of a single frame from the base of `buffer`: the last byte `layout` addresses over
    //!        every plane, so it folds in ImageLayout::planeOffsetBytes and is a frame stride only
    //!        where that offset is zero, as it is for kRGB8.
    //! @return Byte count, or 0 for kNV12BL, whose pixels are sampled rather than addressed.
    int64_t frameBytes() const noexcept;

    //! @brief The same reach over every plane and all `frames`.
    //! @return Byte count, or 0 for kNV12BL, whose pixels are sampled rather than addressed.
    int64_t addressedBytes() const noexcept;

    //! @brief Metadata-only copy at a new spatial size: keeps channels, frames and fps, replaces
    //!        height/width, and carries no pixel buffer (the resized pixels live in the caller's device
    //!        tensor).
    //! @return ImageData with the new dimensions and this object's channels, frames and fps.
    ImageData resizedMeta(int64_t newHeight, int64_t newWidth) const;
};

/*!
 * @brief Decode an image file into pinned host RGB
 * @param path Path to image file
 * @return Loaded image data
 * @throws std::runtime_error if image cannot be loaded from file, or memory allocation fails
 */
ImageData loadRgbImageFromFile(std::string const& path);

/*!
 * @brief Decode encoded image bytes into pinned host RGB
 * @param data Pointer to encoded image data in memory
 * @param size Size of encoded image data in bytes
 * @return Loaded image data
 * @throws std::runtime_error if image cannot be loaded from memory, or memory allocation fails
 */
ImageData loadRgbImageFromEncodedBytes(unsigned char const* data, size_t size);

/*!
 * @brief Decode identically-sized image files and stack them into a single 4D `[T, H, W, C]` host RGB tensor
 * @param framePaths One file path per video frame (in temporal order)
 * @param fps Source frame rate used by the runner to compute MRoPE timestamps
 * @return Loaded video as a single ImageData with `frames == framePaths.size()`
 * @throws std::runtime_error if framePaths is empty, any frame fails to load, frame sizes mismatch, or memory
 * allocation fails
 */
ImageData loadRgbVideoFromFrames(std::vector<std::string> const& framePaths, double fps = 1.0);

/*!
 * @brief Wrap pixel storage the caller already owns. Decodes nothing, copies nothing.
 *
 * The buffer shall be device memory or page-locked host memory, since preprocessing reads it in
 * place, and shall outlive the preprocessing stream. kRGB8 shall be tightly packed; kNV12PL may be
 * padded and states its pitchBytes, plus frameStrideBytes when frames > 1; kNV12BL goes through
 * wrapImageTexture. Sets channels = 3, and isVideo when `isVideo` says so or `frames > 1`.
 *
 * @throws std::runtime_error if the buffer, the layout or the extents do not hold to the above, or
 *         the layout reaches past buffer->getMemoryCapacity().
 */
ImageData wrapImageBuffer(std::shared_ptr<rt::Tensor> buffer, ImageLayout layout, int64_t width, int64_t height,
    int64_t frames = 1, bool isVideo = false);

/*!
 * @brief Wrap one block-linear NV12 frame the caller has already imported into CUDA.
 *
 * Imports nothing and owns nothing; the objects and their arrays shall outlive the preprocessing
 * stream. `luma` samples one uint8 per texel over `width x height`, `chroma` an interleaved CbCr pair
 * over `((width + 1) / 2) x ((height + 1) / 2)`. Both shall be plain 2D arrays read with point
 * filtering, cudaReadModeElementType and unnormalised coordinates, sRGB off; the address mode is
 * free, since the kernel clamps its taps.
 *
 * @throws std::runtime_error if a plane does not hold to the above, or the colour metadata is
 *         unspecified.
 */
ImageData wrapImageTexture(cudaTextureObject_t luma, cudaTextureObject_t chroma, ColorStandard colorStandard,
    ColorRange colorRange, int64_t width, int64_t height);

/*!
 * @brief Resize `image` frames `firstFrame ..< firstFrame + numFrames`, convert them to RGB and
 *        apply (v / 255 - mean) / std per channel, reading `image` in place and allocating nothing.
 *
 * `image.layout` supplies the colour standard and range for YUV, and plane addressing for the formats
 * read out of a buffer; kNV12BL is sampled from `image.textures` instead. `dstImage` is reshaped to
 * `[numFrames, outHeight, outWidth, 3]` HALF and must own that much memory. `stream` is drained
 * before an exception leaves, so the caller may release the source frames from its catch.
 *
 * @throws std::runtime_error if `image` carries no pixels in the form its format calls for, the frame
 *         range is outside `image` or reaches past its buffer, an extent is non-positive, a plane
 *         pitch is below its valid row width, `std` has a zero component, `dstImage` is not GPU HALF
 *         or is too small, or the frame exceeds the kernel's 32-bit addressing.
 */
void resizeAndNormalizeToRgb(ImageData const& image, int64_t const firstFrame, int64_t const numFrames,
    std::array<float, 3> const& mean, std::array<float, 3> const& std, rt::Tensor& dstImage, int64_t const outHeight,
    int64_t const outWidth, cudaStream_t stream);

//! @brief Interpolation filter for :func:`resizeImage`.
enum class InterpolationMode
{
    kLINEAR,  //!< Bilinear
    kBICUBIC, //!< Catmull-Rom cubic
};

/*!
 * @brief Resize each frame of an image/video stack into a pre-allocated buffer, unless `image` is already at
 *        the target dimensions.
 * @param image Source image (4D `[T, H, W, C]`)
 * @param resizedImage Output buffer (reshaped to target dimensions); untouched when the resize is skipped
 * @param newWidth Target width
 * @param newHeight Target height
 * @param mode Interpolation filter
 * @return Reference to `image` when its dimensions already match (resize skipped), otherwise reference to the
 *         freshly resized `resizedImage`. Always consume the returned reference, not `resizedImage`.
 * @throws std::runtime_error if `image` is not host-resident kRGB8, or the output buffer cannot be
 *         reshaped
 */
[[nodiscard]] ImageData const& resizeImage(
    ImageData const& image, ImageData& resizedImage, int64_t newWidth, int64_t newHeight, InterpolationMode mode);

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm
