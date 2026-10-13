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

#include "runtime/imageUtils.h"
#include "common/checkMacros.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

namespace
{

//! Left-sited 4:2:0 (MPEG-2 / H.264 / HEVC, so NVDEC and camera ISP output) places a chroma sample at
//! luma column 2j + 0.5, a quarter of a chroma sample left of the grid the filter samples by default.
//! Its chroma rows are interstitial, so only the horizontal axis carries a shift.
constexpr float kChromaShiftLeftSited = 0.25F;

//! Derived from the two luma weights that define the standard: Kg = 1 - Kr - Kb, and the chroma
//! excursion S is 224 codes for limited range and 255 for full.
kernel::YuvToRgbCoeffs yuvToRgbCoeffs(ColorStandard const standard, ColorRange const range)
{
    float kr = 0.0F;
    float kb = 0.0F;
    switch (standard)
    {
    case ColorStandard::kBt601:
        kr = 0.299F;
        kb = 0.114F;
        break;
    case ColorStandard::kBt709:
        kr = 0.2126F;
        kb = 0.0722F;
        break;
    case ColorStandard::kBt2020:
        kr = 0.2627F;
        kb = 0.0593F;
        break;
    default: ELLM_CHECK(false, "yuvToRgbCoeffs: unknown ColorStandard.");
    }

    bool const limited = range == ColorRange::kLimited;
    float const kg = 1.0F - kr - kb;
    float const s = limited ? 224.0F : 255.0F;

    kernel::YuvToRgbCoeffs coeffs{};
    coeffs.yScale = limited ? 255.0F / 219.0F : 1.0F;
    coeffs.yOffset = limited ? 16.0F : 0.0F;
    coeffs.crToR = 255.0F * 2.0F * (1.0F - kr) / s;
    coeffs.cbToG = -255.0F * 2.0F * kb * (1.0F - kb) / (kg * s);
    coeffs.crToG = -255.0F * 2.0F * kr * (1.0F - kr) / (kg * s);
    coeffs.cbToB = 255.0F * 2.0F * (1.0F - kb) / s;
    return coeffs;
}

//! A host buffer reaching here is page-locked, from cudaMallocHost or a checked wrap. Under UVA its
//! device mapping is the identity, but asking for it turns a missing mapping into an error, not a fault.
unsigned char* deviceAddress(rt::Tensor& buffer)
{
    void* const host = buffer.rawPointer();
    if (buffer.getDeviceType() == rt::DeviceType::kGPU)
    {
        return static_cast<unsigned char*>(host);
    }

    void* mapped = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&mapped, host, 0));
    return static_cast<unsigned char*>(mapped);
}

kernel::SourceFormat sourceFormat(ImageFormat const format)
{
    switch (format)
    {
    case ImageFormat::kRGB8: return kernel::SourceFormat::kRGB8;
    case ImageFormat::kNV12PL: return kernel::SourceFormat::kNV12PL;
    case ImageFormat::kNV12BL: return kernel::SourceFormat::kNV12BL;
    }
    ELLM_CHECK(false, "sourceFormat: unknown ImageFormat.");
    return kernel::SourceFormat::kNONE;
}

//! Preprocessing reads the caller's pixels in place, so a host buffer has to be addressable from a
//! kernel. A pageable pointer is not, and reaching one is an illegal access raised at the next
//! synchronisation point, far from the wrap that admitted it.
void checkKernelAddressable(rt::Tensor const& buffer)
{
    if (buffer.getDeviceType() == rt::DeviceType::kGPU)
    {
        return;
    }

    cudaPointerAttributes attributes{};
    cudaError_t const status = cudaPointerGetAttributes(&attributes, buffer.rawPointer());
    if (status != cudaSuccess)
    {
        (void) cudaGetLastError();
    }
    ELLM_CHECK(status == cudaSuccess && attributes.type != cudaMemoryTypeUnregistered,
        "wrapImageBuffer: a host buffer shall be page-locked and mapped; allocate it with cudaMallocHost or "
        "register it with cudaHostRegister and cudaHostRegisterMapped.");
}

//! Checks a caller-created texture object against the plane it stands for. Filtering, a normalised
//! read or sRGB each yield a plausible but wrong image rather than an error; the address mode is left
//! free because the kernel clamps its tap coordinates before sampling.
void checkTexturePlane(cudaTextureObject_t const texture, std::string const& plane, int32_t const chromaBits,
    int64_t const width, int64_t const height)
{
    std::string const what = "wrapImageTexture: the " + plane + " plane ";
    ELLM_CHECK(texture != 0, what + "texture object is unset.");

    cudaResourceDesc resource{};
    CUDA_CHECK(cudaGetTextureObjectResourceDesc(&resource, texture));
    ELLM_CHECK(resource.resType == cudaResourceTypeArray, what + "shall be backed by a CUDA array.");

    cudaChannelFormatDesc channel{};
    cudaExtent extent{};
    unsigned int flags{};
    CUDA_CHECK(cudaArrayGetInfo(&channel, &extent, &flags, resource.res.array.array));
    ELLM_CHECK(channel.f == cudaChannelFormatKindUnsigned && channel.x == 8 && channel.y == chromaBits && channel.z == 0
            && channel.w == 0,
        what + "shall carry unsigned 8-bit samples in " + std::to_string(chromaBits == 0 ? 1 : 2) + " channel(s).");
    ELLM_CHECK(static_cast<int64_t>(extent.width) == width && static_cast<int64_t>(extent.height) == height,
        what + "array is " + std::to_string(extent.width) + "x" + std::to_string(extent.height)
            + " but the frame needs " + std::to_string(width) + "x" + std::to_string(height) + ".");
    // tex2D reads one layer of a plain 2D array; a layered, 3D or cubemap array samples elsewhere.
    ELLM_CHECK(extent.depth == 0, what + "shall be a plain 2D array, not layered or 3D.");
    // A hardware decoder sets this flag on the surfaces it writes, and it constrains the allocation rather than
    // how tex2D reads the array. cuda.h names the bit CUDA_ARRAY3D_VIDEO_ENCODE_DECODE from CUDA 12 on; the
    // runtime API and the CUDA 11.4 headers report it without a name.
    constexpr unsigned int kArrayVideoEncodeDecode = 0x100U;
    ELLM_CHECK((flags & ~kArrayVideoEncodeDecode) == 0,
        what + "shall be a plain 2D array, without layered, cubemap or surface flags; cudaArrayGetInfo reports flags "
            + std::to_string(flags) + ".");

    cudaTextureDesc sampler{};
    CUDA_CHECK(cudaGetTextureObjectTextureDesc(&sampler, texture));
    ELLM_CHECK(sampler.filterMode == cudaFilterModePoint, what + "shall sample with cudaFilterModePoint.");
    ELLM_CHECK(sampler.readMode == cudaReadModeElementType, what + "shall read with cudaReadModeElementType.");
    ELLM_CHECK(sampler.normalizedCoords == 0, what + "shall use unnormalised coordinates.");
    ELLM_CHECK(sampler.sRGB == 0, what + "shall not apply the sRGB transfer function.");
}

//! Valid bytes per row and rows per frame, both fixed by the format and the frame size.
struct PlaneExtent
{
    int64_t validRowBytes;
    int64_t rows;
};

PlaneExtent planeExtent(ImageFormat const format, int64_t const width, int64_t const height, int32_t const plane)
{
    bool const isNv12 = format != ImageFormat::kRGB8;
    if (plane == 0)
    {
        return {isNv12 ? width : width * 3, height};
    }
    return {2 * ((width + 1) / 2), (height + 1) / 2};
}

//! Bytes one frame of `plane` occupies at the layout's pitch.
int64_t planeFrameBytes(ImageLayout const& layout, int64_t const width, int64_t const height, int32_t const plane)
{
    PlaneExtent const extent = planeExtent(layout.format, width, height, plane);
    return (extent.rows - 1) * layout.pitchBytes[plane] + extent.validRowBytes;
}

//! Last byte the layout addresses. A single frame ignores the frame stride, which wrapImageBuffer
//! leaves unchecked for it.
int64_t layoutAddressedBytes(ImageLayout const& layout, int64_t const width, int64_t const height, int64_t const frames)
{
    // Block-linear planes are sampled through textures, so the layout's pitches and offsets do not
    // describe them and no byte of a buffer is reached.
    if (layout.format == ImageFormat::kNV12BL)
    {
        return 0;
    }

    int32_t const numPlanes = layout.format == ImageFormat::kRGB8 ? 1 : 2;
    int64_t last = 0;
    for (int32_t p = 0; p < numPlanes; ++p)
    {
        int64_t const frameStride = frames > 1 ? layout.frameStrideBytes[p] : 0;
        last = std::max(
            last, layout.planeOffsetBytes[p] + (frames - 1) * frameStride + planeFrameBytes(layout, width, height, p));
    }
    return last;
}

} // namespace

int64_t ImageData::frameBytes() const noexcept
{
    return layoutAddressedBytes(layout, width, height, 1);
}

int64_t ImageData::addressedBytes() const noexcept
{
    return layoutAddressedBytes(layout, width, height, frames);
}

ImageData::ImageData(rt::Tensor&& data)
{
    check::check(data.getDataType() == nvinfer1::DataType::kUINT8, "Image data must be UINT8");
    check::check(data.getShape().getNumDims() == 4, "Image data must be 4D [T, H, W, C]");
    check::check(data.getShape()[3] == 3, "Image data must have 3 channels");

    // [frames, height, width, channels]
    frames = data.getShape()[0];
    height = data.getShape()[1];
    width = data.getShape()[2];
    channels = data.getShape()[3];
    check::check(frames > 0, "Image frame sequence must have frames > 0");
    buffer = std::make_shared<rt::Tensor>(std::move(data));

    layout.format = ImageFormat::kRGB8;
    layout.pitchBytes[0] = width * channels;
    layout.frameStrideBytes[0] = frameBytes();
}

unsigned char* ImageData::data() const noexcept
{
    if (!buffer || layout.format != ImageFormat::kRGB8 || buffer->getDeviceType() != rt::DeviceType::kCPU)
    {
        return nullptr;
    }
    return buffer->dataPointer<unsigned char>();
}

ImageData ImageData::resizedMeta(int64_t newHeight, int64_t newWidth) const
{
    ImageData meta{};
    meta.height = newHeight;
    meta.width = newWidth;
    meta.channels = channels;
    meta.frames = frames;
    meta.fps = fps;
    // The pixels described are the packed RGB the pipeline produces, whatever the source layout was.
    meta.layout.format = ImageFormat::kRGB8;
    meta.layout.pitchBytes[0] = newWidth * channels;
    meta.layout.frameStrideBytes[0] = meta.frameBytes();
    return meta;
}

ImageData loadRgbImageFromFile(std::string const& path)
{
    int width{0}, height{0}, channels{0};
    int desiredChannels = 3;
    unsigned char* imageData = stbi_load(path.c_str(), &width, &height, &channels, desiredChannels);
    ELLM_CHECK(imageData != nullptr, "Failed to load image: " + path + " - " + std::string(stbi_failure_reason()));

    rt::Tensor imgTensor{};
    // Need to handle the logic where space allocation for image tensor failed. We need to free the image data and
    // throw an exception.
    try
    {
        imgTensor = rt::Tensor({1, height, width, desiredChannels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8,
            "imageUtils::loadRgbImageFromFile::imgTensor");
    }
    catch (std::exception const& e)
    {
        stbi_image_free(imageData);
        throw std::runtime_error("Failed to allocate space for image tensor: " + std::string(e.what()));
    }
    memcpy(imgTensor.dataPointer<unsigned char>(), imageData, width * height * desiredChannels);
    stbi_image_free(imageData);
    return ImageData(std::move(imgTensor));
}

ImageData loadRgbImageFromEncodedBytes(unsigned char const* data, size_t size)
{
    int width{0}, height{0}, channels{0};
    int desiredChannels = 3;
    unsigned char* imageData = stbi_load_from_memory(data, size, &width, &height, &channels, desiredChannels);
    ELLM_CHECK(imageData != nullptr, "Failed to load image from memory: " + std::string(stbi_failure_reason()));

    rt::Tensor imgTensor{};
    // Need to handle the logic where space allocation for image tensor failed. We need to free the image data and
    // throw an exception.
    try
    {
        imgTensor = rt::Tensor({1, height, width, desiredChannels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8,
            "imageUtils::loadRgbImageFromEncodedBytes::imgTensor");
    }
    catch (std::exception const& e)
    {
        stbi_image_free(imageData);
        throw std::runtime_error("Failed to allocate space for image tensor: " + std::string(e.what()));
    }
    memcpy(imgTensor.dataPointer<unsigned char>(), imageData, width * height * desiredChannels);
    stbi_image_free(imageData);
    return ImageData(std::move(imgTensor));
}

ImageData loadRgbVideoFromFrames(std::vector<std::string> const& framePaths, double const fps)
{
    ELLM_CHECK(!framePaths.empty(), "loadRgbVideoFromFrames: framePaths is empty");
    ELLM_CHECK(std::isfinite(fps) && fps > 0.0,
        "loadRgbVideoFromFrames: fps must be a positive finite number, got " + std::to_string(fps));

    // Load the first frame to determine the common (H, W, C).
    ImageData firstFrame = loadRgbImageFromFile(framePaths[0]);
    int64_t const T = static_cast<int64_t>(framePaths.size());
    int64_t const H = firstFrame.height;
    int64_t const W = firstFrame.width;
    int64_t const C = firstFrame.channels;
    int64_t const frameBytes = firstFrame.frameBytes();

    rt::Tensor stacked{};
    try
    {
        stacked = rt::Tensor({T, H, W, C}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8,
            "imageUtils::loadRgbVideoFromFrames::stacked");
    }
    catch (std::exception const& e)
    {
        throw std::runtime_error("Failed to allocate space for video tensor: " + std::string(e.what()));
    }
    unsigned char* dst = stacked.dataPointer<unsigned char>();
    std::memcpy(dst, firstFrame.data(), frameBytes);

    for (int64_t t = 1; t < T; ++t)
    {
        ImageData frame = loadRgbImageFromFile(framePaths[t]);
        ELLM_CHECK(frame.height == H && frame.width == W && frame.channels == C,
            "loadRgbVideoFromFrames: frame " + std::to_string(t) + " has shape (" + std::to_string(frame.height) + ", "
                + std::to_string(frame.width) + ", " + std::to_string(frame.channels) + ") but expected ("
                + std::to_string(H) + ", " + std::to_string(W) + ", " + std::to_string(C) + ")");
        std::memcpy(dst + t * frameBytes, frame.data(), frameBytes);
    }

    ImageData video(std::move(stacked));
    video.fps = fps;
    video.isVideo = true;
    return video;
}

ImageData const& resizeImage(
    ImageData const& image, ImageData& resizedImage, int64_t newWidth, int64_t newHeight, InterpolationMode mode)
{
    // stbir reads and writes tightly packed host RGB, which is what data() guarantees.
    ELLM_CHECK(image.layout.format == ImageFormat::kRGB8,
        "resizeImage: this CPU path takes packed RGB8; NV12 goes through the GPU preprocessing path.");
    ELLM_CHECK(image.data() != nullptr, "resizeImage: source pixels are not readable as host RGB.");
    ELLM_CHECK(resizedImage.buffer != nullptr && resizedImage.buffer->getDeviceType() == rt::DeviceType::kCPU,
        "resizeImage: the destination shall carry a host buffer for stbir to write.");

    // Already at the target size — skip the resample and the scratch-buffer copy.
    if (image.width == newWidth && image.height == newHeight)
    {
        return image;
    }

    // Reshape pre-allocated buffer to target [T, H, W, C] (always 4D).
    bool const success = resizedImage.buffer->reshape({image.frames, newHeight, newWidth, image.channels});
    ELLM_CHECK(success, "Failed to reshape resized image buffer");
    resizedImage.frames = image.frames;
    resizedImage.height = newHeight;
    resizedImage.width = newWidth;
    resizedImage.channels = image.channels;
    resizedImage.fps = image.fps; // preserve sample fps (drives Qwen2.5-VL video MRoPE time interval)
    resizedImage.layout.format = ImageFormat::kRGB8;
    resizedImage.layout.pitchBytes[0] = newWidth * image.channels;
    resizedImage.layout.frameStrideBytes[0] = resizedImage.frameBytes();

    // Resize the image(s) into the pre-allocated buffer. stbir is invoked once per frame.
    constexpr int32_t kINPUT_STRIDE_BYTES{0};
    constexpr int32_t kOUTPUT_STRIDE_BYTES{0};
    int64_t const srcFrameBytes = image.frameBytes();
    int64_t const dstFrameBytes = resizedImage.frameBytes();
    for (int64_t t = 0; t < image.frames; ++t)
    {
        unsigned char const* src = image.data() + t * srcFrameBytes;
        unsigned char* dst = resizedImage.data() + t * dstFrameBytes;
        if (mode == InterpolationMode::kBICUBIC)
        {
            stbir_resize(src, image.width, image.height, kINPUT_STRIDE_BYTES, dst, newWidth, newHeight,
                kOUTPUT_STRIDE_BYTES, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM);
        }
        else
        {
            stbir_resize_uint8_linear(src, image.width, image.height, kINPUT_STRIDE_BYTES, dst, newWidth, newHeight,
                kOUTPUT_STRIDE_BYTES, STBIR_RGB);
        }
    }
    return resizedImage;
}

ImageData wrapImageBuffer(
    std::shared_ptr<rt::Tensor> buffer, ImageLayout layout, int64_t width, int64_t height, int64_t frames, bool isVideo)
{
    ELLM_CHECK(buffer != nullptr, "wrapImageBuffer: buffer is null.");
    ELLM_CHECK(buffer->getDataType() == nvinfer1::DataType::kUINT8, "wrapImageBuffer: buffer shall be UINT8.");
    ELLM_CHECK(width > 0 && height > 0 && frames > 0, "wrapImageBuffer: dimensions shall be positive.");
    ELLM_CHECK(layout.format != ImageFormat::kNV12BL,
        "wrapImageBuffer: kNV12BL addresses no buffer; wrap its texture planes with wrapImageTexture.");

    checkKernelAddressable(*buffer);

    bool const isNv12 = layout.format == ImageFormat::kNV12PL;
    ELLM_CHECK(!isNv12
            || (layout.colorStandard != ColorStandard::kUnspecified && layout.colorRange != ColorRange::kUnspecified),
        "wrapImageBuffer: kNV12PL requires an explicit colour standard and range.");

    // kRGB8 consumers read the whole buffer through ImageData::data() as one contiguous run.
    if (!isNv12)
    {
        ELLM_CHECK(layout.planeOffsetBytes[0] == 0 && layout.pitchBytes[0] == width * 3
                && (frames == 1 || layout.frameStrideBytes[0] == height * width * 3),
            "wrapImageBuffer: kRGB8 shall be tightly packed (plane offset 0, pitch 3 * width, frame stride "
            "height * 3 * width); pitched RGB is not supported.");
    }

    int32_t const numPlanes = isNv12 ? 2 : 1;
    for (int32_t p = 0; p < numPlanes; ++p)
    {
        ELLM_CHECK(layout.planeOffsetBytes[p] >= 0, "wrapImageBuffer: negative plane offset.");
        int64_t const validRowBytes = planeExtent(layout.format, width, height, p).validRowBytes;
        ELLM_CHECK(layout.pitchBytes[p] >= validRowBytes,
            "wrapImageBuffer: plane " + std::to_string(p) + " pitch " + std::to_string(layout.pitchBytes[p])
                + " is below its valid row width " + std::to_string(validRowBytes) + "; supply the pitch explicitly.");

        int64_t const planeBytes = planeFrameBytes(layout, width, height, p);
        if (frames > 1)
        {
            ELLM_CHECK(layout.frameStrideBytes[p] >= planeBytes,
                "wrapImageBuffer: plane " + std::to_string(p) + " frame stride "
                    + std::to_string(layout.frameStrideBytes[p]) + " is below one frame of that plane ("
                    + std::to_string(planeBytes) + " bytes); supply the frame stride explicitly.");
        }
    }
    int64_t const lastByte = layoutAddressedBytes(layout, width, height, frames);
    ELLM_CHECK(lastByte <= buffer->getMemoryCapacity(),
        "wrapImageBuffer: layout addresses " + std::to_string(lastByte) + " bytes but the buffer holds "
            + std::to_string(buffer->getMemoryCapacity()) + ".");

    ImageData image{};
    image.buffer = std::move(buffer);
    image.layout = layout;
    image.width = width;
    image.height = height;
    image.channels = 3;
    image.frames = frames;
    image.isVideo = isVideo || frames > 1;
    return image;
}

ImageData wrapImageTexture(cudaTextureObject_t const luma, cudaTextureObject_t const chroma,
    ColorStandard const colorStandard, ColorRange const colorRange, int64_t const width, int64_t const height)
{
    ELLM_CHECK(width > 0 && height > 0, "wrapImageTexture: dimensions shall be positive.");
    ELLM_CHECK(colorStandard != ColorStandard::kUnspecified && colorRange != ColorRange::kUnspecified,
        "wrapImageTexture: kNV12BL requires an explicit colour standard and range.");

    checkTexturePlane(luma, "luma", 0, width, height);
    checkTexturePlane(chroma, "chroma", 8, (width + 1) / 2, (height + 1) / 2);

    ImageData image{};
    image.textures.plane0 = luma;
    image.textures.plane1 = chroma;
    image.layout.format = ImageFormat::kNV12BL;
    image.layout.colorStandard = colorStandard;
    image.layout.colorRange = colorRange;
    image.width = width;
    image.height = height;
    image.channels = 3;
    return image;
}

void resizeAndNormalizeToRgb(ImageData const& image, int64_t const firstFrame, int64_t const numFrames,
    std::array<float, 3> const& mean, std::array<float, 3> const& std, rt::Tensor& dstImage, int64_t const outHeight,
    int64_t const outWidth, cudaStream_t stream)
{
    // Validation runs while an earlier call's kernels may still be reading the caller's frames on this
    // stream, so it sits inside the try as well: an exception drains the stream before it leaves.
    try
    {
        bool const isBlockLinear = image.layout.format == ImageFormat::kNV12BL;
        if (isBlockLinear)
        {
            ELLM_CHECK(image.textures.plane0 != 0 && image.textures.plane1 != 0,
                "resizeAndNormalizeToRgb: a kNV12BL image carries no texture planes; wrap it with wrapImageTexture.");
        }
        else
        {
            ELLM_CHECK(image.buffer != nullptr, "resizeAndNormalizeToRgb: image buffer is null.");
        }
        ELLM_CHECK(firstFrame >= 0 && numFrames > 0 && firstFrame + numFrames <= image.frames,
            "resizeAndNormalizeToRgb: frames [" + std::to_string(firstFrame) + ", "
                + std::to_string(firstFrame + numFrames) + ") lie outside the " + std::to_string(image.frames)
                + " the image holds.");
        ELLM_CHECK(image.height > 0 && image.width > 0 && outHeight > 0 && outWidth > 0,
            "resizeAndNormalizeToRgb: frame dimensions shall be positive.");
        if (!isBlockLinear)
        {
            int32_t const numPlanes = image.layout.format == ImageFormat::kRGB8 ? 1 : 2;
            for (int32_t p = 0; p < numPlanes; ++p)
            {
                int64_t const validRowBytes
                    = planeExtent(image.layout.format, image.width, image.height, p).validRowBytes;
                ELLM_CHECK(image.layout.pitchBytes[p] >= validRowBytes,
                    "resizeAndNormalizeToRgb: plane " + std::to_string(p) + " pitch "
                        + std::to_string(image.layout.pitchBytes[p]) + " is below its valid row width "
                        + std::to_string(validRowBytes) + ".");
            }
            // `layout` is a public member of an open aggregate, so a frame range that fits when the
            // image was wrapped can be widened afterwards. The kernel reads the caller's buffer in
            // place, so re-derive the reach here rather than trust the wrap.
            int64_t const lastByte
                = layoutAddressedBytes(image.layout, image.width, image.height, firstFrame + numFrames);
            ELLM_CHECK(lastByte <= image.buffer->getMemoryCapacity(),
                "resizeAndNormalizeToRgb: the layout reaches " + std::to_string(lastByte)
                    + " bytes but the buffer holds " + std::to_string(image.buffer->getMemoryCapacity()) + ".");
        }
        ELLM_CHECK(std[0] != 0.0F && std[1] != 0.0F && std[2] != 0.0F,
            "resizeAndNormalizeToRgb: every std component shall be non-zero.");
        ELLM_CHECK(dstImage.getDeviceType() == rt::DeviceType::kGPU,
            "resizeAndNormalizeToRgb: destination shall be on the GPU.");
        ELLM_CHECK(
            dstImage.getDataType() == nvinfer1::DataType::kHALF, "resizeAndNormalizeToRgb: destination shall be HALF.");
        ELLM_CHECK(dstImage.reshape({numFrames, outHeight, outWidth, 3}),
            "resizeAndNormalizeToRgb: destination too small for " + std::to_string(numFrames) + " frames of "
                + std::to_string(outHeight) + "x" + std::to_string(outWidth) + " RGB.");

        // The kernel addresses source and destination with int, and narrows both plane pitches to it.
        // A block-linear source is reached by texture coordinate rather than by offset, so its extent
        // is the bound.
        int64_t const intMax = std::numeric_limits<int>::max();
        int64_t const srcBound = isBlockLinear
            ? std::max(image.width, image.height)
            : std::max(image.height * image.layout.pitchBytes[0], image.layout.pitchBytes[1]);
        ELLM_CHECK(outHeight * outWidth * 3 <= intMax && srcBound <= intMax && outWidth * 3 <= intMax,
            "resizeAndNormalizeToRgb: a frame of " + std::to_string(image.height) + "x" + std::to_string(image.width)
                + " to " + std::to_string(outHeight) + "x" + std::to_string(outWidth)
                + " exceeds the kernel's 32-bit addressing.");

        bool const isYuv = image.layout.format != ImageFormat::kRGB8;
        ELLM_CHECK(!isYuv
                || (image.layout.colorStandard != ColorStandard::kUnspecified
                    && image.layout.colorRange != ColorRange::kUnspecified),
            "resizeAndNormalizeToRgb: a YUV source shall name a colour standard and range.");
        kernel::YuvToRgbCoeffs const coeffs
            = isYuv ? yuvToRgbCoeffs(image.layout.colorStandard, image.layout.colorRange) : kernel::YuvToRgbCoeffs{};

        unsigned char* const base = isBlockLinear ? nullptr : deviceAddress(*image.buffer);
        auto* const dstBase = dstImage.dataPointer<half>();
        int64_t const outFrameElems = outHeight * outWidth * 3;

        // The batch dimension reaches frame f at f * height rows into the plane, which a frame stride
        // wider than the plane's footprint does not satisfy, and indexes the destination in int. YUV
        // is excluded because its two planes advance by different row counts per frame.
        bool const framesStackVertically = !isYuv
            && image.layout.frameStrideBytes[0] == image.height * image.layout.pitchBytes[0]
            && numFrames * outFrameElems <= intMax;
        int64_t const batchSize = framesStackVertically ? numFrames : 1;

        for (int64_t t = 0; t < numFrames; t += batchSize)
        {
            int64_t const src = firstFrame + t;
            void const* plane0 = nullptr;
            void const* plane1 = nullptr;
            if (isBlockLinear)
            {
                // For kNV12BL the kernel reads both planes as texture objects in these parameters.
                plane0 = reinterpret_cast<void const*>(static_cast<uintptr_t>(image.textures.plane0));
                plane1 = reinterpret_cast<void const*>(static_cast<uintptr_t>(image.textures.plane1));
            }
            else
            {
                plane0 = base + image.layout.planeOffsetBytes[0] + src * image.layout.frameStrideBytes[0];
                plane1 = isYuv ? base + image.layout.planeOffsetBytes[1] + src * image.layout.frameStrideBytes[1]
                               : nullptr;
            }

            kernel::batchedPreprocessImage(plane0, plane1, static_cast<int>(image.width),
                static_cast<int>(image.layout.pitchBytes[0]), static_cast<int>(image.height),
                static_cast<int>(batchSize), sourceFormat(image.layout.format),
                static_cast<int>(image.layout.pitchBytes[1]), kChromaShiftLeftSited, coeffs,
                dstBase + t * outFrameElems, static_cast<int>(outWidth), static_cast<int>(outWidth * 3),
                static_cast<int>(outHeight), kernel::PixelDataType::kHALF, kernel::PixelLayout::kNHWC_RGB,
                kernel::Interpolation::kBICUBIC, mean[0], mean[1], mean[2], std[0], std[1], std[2], stream);
        }
    }
    catch (...)
    {
        // The caller owns the source frames and may release them once this returns.
        (void) cudaStreamSynchronize(stream);
        throw;
    }
}

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm
