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

//! Maps an NvSciBufObj or NvBufSurface NV12 frame to the CUDA rt::imageUtils::ImageData the runtime reads.

#pragma once

#include "message.h"

#include "common/tensor.h"
#include "runtime/imageUtils.h"

#include <cuda_runtime_api.h>

#include <nvscibuf.h>

#if EDGELLM_IPC_HAS_NVBUFSURFACE
#include <nvbufsurface.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace edgellm::ipc
{

namespace rt = trt_edgellm::rt;

inline constexpr size_t kMaxImagePlanes = 2;

inline constexpr int64_t kNvSciBufImportTimeoutUs = 1000000;

inline void checkCuda(cudaError_t status, char const* what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(status));
    }
}

namespace detail
{

//! An attribute the list never stated leaves @p out alone, which is what the defaults rest on.
template <typename T>
void readNvSciBufAttr(NvSciBufAttrKeyValuePair const& pair, size_t count, T* out)
{
    if (pair.value != nullptr)
    {
        std::memcpy(out, pair.value, std::min(pair.len, count * sizeof(T)));
    }
}

//! Point filtering and element type, which is what wrapImageTexture requires of both planes.
inline cudaTextureObject_t createPlaneTexture(cudaArray_t array)
{
    cudaResourceDesc resource = {};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;

    cudaTextureDesc sampling = {};
    sampling.addressMode[0] = cudaAddressModeClamp;
    sampling.addressMode[1] = cudaAddressModeClamp;
    sampling.filterMode = cudaFilterModePoint;
    sampling.readMode = cudaReadModeElementType;

    cudaTextureObject_t object = 0;
    checkCuda(cudaCreateTextureObject(&object, &resource, &sampling, nullptr), "cudaCreateTextureObject");
    return object;
}

inline void destroyPlaneTextures(cudaTextureObject_t (&textures)[kMaxImagePlanes]) noexcept
{
    for (cudaTextureObject_t& texture : textures)
    {
        if (texture != 0)
        {
            cudaDestroyTextureObject(texture);
            texture = 0;
        }
    }
}

inline rt::imageUtils::ImageLayout nv12Layout(bool blockLinear) noexcept
{
    rt::imageUtils::ImageLayout layout{};
    layout.format = blockLinear ? rt::imageUtils::ImageFormat::kNV12BL : rt::imageUtils::ImageFormat::kNV12PL;
    layout.colorStandard = rt::imageUtils::ColorStandard::kBt601;
    layout.colorRange = rt::imageUtils::ColorRange::kLimited;
    return layout;
}

//! An unset color standard reads back as 0 (NvSciColorStd_SRGB), which keeps the default BT.601 limited range.
inline void applyColor(rt::imageUtils::ImageLayout& layout, NvSciBufAttrValColorStd standard) noexcept
{
    switch (standard)
    {
    case NvSciColorStd_REC709_SR: layout.colorStandard = rt::imageUtils::ColorStandard::kBt709; break;
    case NvSciColorStd_REC709_ER:
        layout.colorStandard = rt::imageUtils::ColorStandard::kBt709;
        layout.colorRange = rt::imageUtils::ColorRange::kFull;
        break;
    case NvSciColorStd_REC2020_SR: layout.colorStandard = rt::imageUtils::ColorStandard::kBt2020; break;
    case NvSciColorStd_REC2020_ER:
        layout.colorStandard = rt::imageUtils::ColorStandard::kBt2020;
        layout.colorRange = rt::imageUtils::ColorRange::kFull;
        break;
    case NvSciColorStd_REC601_ER: layout.colorRange = rt::imageUtils::ColorRange::kFull; break;
    default: break;
    }
}

#if EDGELLM_IPC_HAS_NVBUFSURFACE
inline void applyColor(rt::imageUtils::ImageLayout& layout, NvBufSurfaceColorFormat format) noexcept
{
    switch (format)
    {
    case NVBUF_COLOR_FORMAT_NV12_ER: layout.colorRange = rt::imageUtils::ColorRange::kFull; break;
    case NVBUF_COLOR_FORMAT_NV12_709: layout.colorStandard = rt::imageUtils::ColorStandard::kBt709; break;
    case NVBUF_COLOR_FORMAT_NV12_709_ER:
        layout.colorStandard = rt::imageUtils::ColorStandard::kBt709;
        layout.colorRange = rt::imageUtils::ColorRange::kFull;
        break;
    case NVBUF_COLOR_FORMAT_NV12_2020: layout.colorStandard = rt::imageUtils::ColorStandard::kBt2020; break;
    default: break;
    }
}
#endif

inline rt::imageUtils::ImageData addressableFrame(
    void* base, uint64_t sizeBytes, rt::imageUtils::ImageLayout const& layout, uint32_t width, uint32_t height)
{
    auto buffer = std::make_shared<rt::Tensor>(base, rt::Coords{static_cast<int64_t>(sizeBytes)}, rt::DeviceType::kGPU,
        trt_edgellm::DataType::kUINT8, "ipcNv12Frame");

    return rt::imageUtils::wrapImageBuffer(std::move(buffer), layout, width, height);
}

} // namespace detail

// --- NvSciBufObj -> CUDA (TensorRT Edge-LLM runtime buffer) ---------------------------------------------------------

inline rt::imageUtils::ImageData mapNvSciBufToCuda(
    NvSciBufObj obj, cudaExternalMemory_t& memory, cudaMipmappedArray_t* mipmaps)
{
    NvSciBufAttrList attrList = nullptr;
    NvSciBufObjGetAttrList(obj, &attrList);
    // The order here is the only thing tying a key to where its answer lands.
    NvSciBufAttrKeyValuePair pairs[] = {
        {NvSciBufImageAttrKey_Layout, nullptr, 0},
        {NvSciBufImageAttrKey_Size, nullptr, 0},
        {NvSciBufImageAttrKey_PlaneCount, nullptr, 0},
        {NvSciBufImageAttrKey_PlaneWidth, nullptr, 0},
        {NvSciBufImageAttrKey_PlaneHeight, nullptr, 0},
        {NvSciBufImageAttrKey_PlanePitch, nullptr, 0},
        {NvSciBufImageAttrKey_PlaneOffset, nullptr, 0},
        {NvSciBufImageAttrKey_PlaneColorStd, nullptr, 0},
    };
    NvSciBufAttrListGetAttrs(attrList, pairs, sizeof(pairs) / sizeof(pairs[0]));
    NvSciBufAttrValImageLayoutType layoutType{};
    uint64_t sizeBytes = 0;
    uint32_t planeCount = 0;
    uint32_t widths[kMaxImagePlanes]{};
    uint32_t heights[kMaxImagePlanes]{};
    uint32_t pitches[kMaxImagePlanes]{};
    uint64_t offsets[kMaxImagePlanes]{};
    NvSciBufAttrValColorStd standard{}; // Plane 0's, which is the frame's.
    detail::readNvSciBufAttr(pairs[0], 1, &layoutType);
    detail::readNvSciBufAttr(pairs[1], 1, &sizeBytes);
    detail::readNvSciBufAttr(pairs[2], 1, &planeCount);
    // The plane arrays here hold two, whatever the buffer reports.
    planeCount = std::min<uint32_t>(planeCount, kMaxImagePlanes);
    detail::readNvSciBufAttr(pairs[3], planeCount, widths);
    detail::readNvSciBufAttr(pairs[4], planeCount, heights);
    detail::readNvSciBufAttr(pairs[5], planeCount, pitches);
    detail::readNvSciBufAttr(pairs[6], planeCount, offsets);
    detail::readNvSciBufAttr(pairs[7], 1, &standard);
    bool const blockLinear = layoutType == NvSciBufImage_BlockLinearType;

    rt::imageUtils::ImageLayout layout = detail::nv12Layout(blockLinear);
    detail::applyColor(layout, standard);

    cudaExternalMemoryHandleDesc memoryDesc = {};
    memoryDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memoryDesc.handle.nvSciBufObject = obj;
    memoryDesc.size = sizeBytes;
    checkCuda(cudaImportExternalMemory(&memory, &memoryDesc), "cudaImportExternalMemory");

    // Freed here if anything below throws; on success they belong to the returned frame.
    void* base = nullptr;
    cudaTextureObject_t planeTextures[kMaxImagePlanes]{};

    try
    {
        if (!blockLinear)
        {
            cudaExternalMemoryBufferDesc bufferDesc = {};
            bufferDesc.offset = 0;
            bufferDesc.size = sizeBytes;
            checkCuda(
                cudaExternalMemoryGetMappedBuffer(&base, memory, &bufferDesc), "cudaExternalMemoryGetMappedBuffer");

            for (uint32_t i = 0; i < planeCount; ++i)
            {
                layout.planeOffsetBytes[i] = static_cast<int64_t>(offsets[i]);
                layout.pitchBytes[i] = static_cast<int64_t>(pitches[i]);
            }

            return detail::addressableFrame(base, sizeBytes, layout, widths[0], heights[0]);
        }

        for (uint32_t i = 0; i < planeCount; ++i)
        {
            // The plane's logical size, not the padded allocation; CUDA derives the block-linear layout from it.
            cudaExtent extent = {};
            extent.width = widths[i];
            extent.height = heights[i];
            extent.depth = 0;

            cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {};
            mipmapDesc.offset = offsets[i];
            mipmapDesc.formatDesc = i == 0 ? cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned)
                                           : cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
            mipmapDesc.extent = extent;
            mipmapDesc.flags = 0;
            mipmapDesc.numLevels = 1;
            checkCuda(cudaExternalMemoryGetMappedMipmappedArray(&mipmaps[i], memory, &mipmapDesc),
                "cudaExternalMemoryGetMappedMipmappedArray");

            cudaArray_t level = nullptr;
            checkCuda(cudaGetMipmappedArrayLevel(&level, mipmaps[i], 0), "cudaGetMipmappedArrayLevel");
            planeTextures[i] = detail::createPlaneTexture(level);
        }

        return rt::imageUtils::wrapImageTexture(
            planeTextures[0], planeTextures[1], layout.colorStandard, layout.colorRange, widths[0], heights[0]);
    }
    catch (...)
    {
        detail::destroyPlaneTextures(planeTextures);
        if (base != nullptr)
        {
            cudaFree(base);
        }
        throw;
    }
}

//! Frees the CUDA mappings before the external memory, as CUDA requires; the NvSciBufObj stays the caller's.
inline void unmapNvSciBufFromCuda(
    cudaExternalMemory_t& memory, cudaMipmappedArray_t* mipmaps, rt::imageUtils::ImageData& frame) noexcept
{
    for (cudaTextureObject_t const texture : {frame.textures.plane0, frame.textures.plane1})
    {
        if (texture != 0)
        {
            cudaDestroyTextureObject(texture);
        }
    }

    // Take the mapped base before clearing the frame: only its tensor holds that pointer.
    void* const base = frame.buffer == nullptr ? nullptr : frame.buffer->rawPointer();
    frame = rt::imageUtils::ImageData{};

    for (size_t i = 0; i < kMaxImagePlanes; ++i)
    {
        if (mipmaps[i] != nullptr)
        {
            cudaFreeMipmappedArray(mipmaps[i]);
            mipmaps[i] = nullptr;
        }
    }
    if (base != nullptr)
    {
        cudaFree(base);
    }
    if (memory != nullptr)
    {
        cudaDestroyExternalMemory(memory);
        memory = nullptr;
    }

    // Clear the CUDA error state so teardown does not surface an error raised elsewhere.
    static_cast<void>(cudaGetLastError());
}

// --- NvBufSurface -> CUDA ImageData for the runtime ------------------------------------------

#if EDGELLM_IPC_HAS_NVBUFSURFACE
inline rt::imageUtils::ImageData mapNvBufSurfaceToCuda(NvBufSurface* surface)
{
    NvBufSurfaceParams const& described = surface->surfaceList[0];
    NvBufSurfacePlaneParams const& planes = described.planeParams;

    NvBufSurfaceMapCudaBuffer(surface, 0);

    auto const* cudaBuffer = static_cast<NvBufSurfaceCudaBuffer const*>(described.mappedAddr.cudaPtr);
    if (cudaBuffer == nullptr)
    {
        throw std::runtime_error("frame buffer reported no CUDA mapping after NvBufSurfaceMapCudaBuffer succeeded");
    }

    bool const blockLinear = described.layout == NVBUF_LAYOUT_BLOCK_LINEAR;

    // Clamp to kMaxImagePlanes: the plane arrays below hold two, whatever the surface reports.
    uint32_t const planeCount = std::min<uint32_t>(planes.num_planes, kMaxImagePlanes);

    rt::imageUtils::ImageLayout layout = detail::nv12Layout(blockLinear);
    detail::applyColor(layout, described.colorFormat);

    if (!blockLinear)
    {
        for (uint32_t i = 0; i < planeCount; ++i)
        {
            layout.planeOffsetBytes[i] = static_cast<int64_t>(planes.offset[i]);
            layout.pitchBytes[i] = static_cast<int64_t>(planes.pitch[i]);
        }

        // Measured on Thor: dataPtr is the base as a device address, and basePtr is null.
        return detail::addressableFrame(
            cudaBuffer->dataPtr, described.dataSize, layout, planes.width[0], planes.height[0]);
    }

    // Block linear: dataPtr is a host pointer, so go through the mipmap and take each plane from its level.
    auto const borrowedMipmap = static_cast<cudaMipmappedArray_t>(cudaBuffer->mipmap);
    cudaArray_t level = nullptr;
    checkCuda(cudaGetMipmappedArrayLevel(&level, borrowedMipmap, 0), "cudaGetMipmappedArrayLevel");

    // Freed here if a texture fails; on success they belong to the returned frame.
    cudaTextureObject_t planeTextures[kMaxImagePlanes]{};
    try
    {
        // One CUDA array holds both planes here, unlike the NvSciBuf route's one array per plane.
        for (uint32_t i = 0; i < planeCount; ++i)
        {
            cudaArray_t plane = nullptr;
            checkCuda(cudaArrayGetPlane(&plane, level, i), "cudaArrayGetPlane");
            planeTextures[i] = detail::createPlaneTexture(plane);
        }
    }
    catch (...)
    {
        detail::destroyPlaneTextures(planeTextures);
        throw;
    }

    return rt::imageUtils::wrapImageTexture(
        planeTextures[0], planeTextures[1], layout.colorStandard, layout.colorRange, planes.width[0], planes.height[0]);
}

//! The surface owns its CUDA mapping; only the textures created by mapNvBufSurfaceToCuda are freed here.
inline void unmapNvBufSurfaceFromCuda(NvBufSurface* surface, rt::imageUtils::ImageData& frame) noexcept
{
    for (cudaTextureObject_t const texture : {frame.textures.plane0, frame.textures.plane1})
    {
        if (texture != 0)
        {
            cudaDestroyTextureObject(texture);
        }
    }
    frame = rt::imageUtils::ImageData{};

    if (surface != nullptr && surface->surfaceList[0].mappedAddr.cudaPtr != nullptr)
    {
        NvBufSurfaceUnMapCudaBuffer(surface, 0);
    }

    static_cast<void>(cudaGetLastError());
}
#endif

} // namespace edgellm::ipc
