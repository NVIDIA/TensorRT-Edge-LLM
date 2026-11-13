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

//! Test data only: the frames and samples a camera and a microphone would already be producing.

#pragma once

#include "ipcClient.h"

#include <nvscibuf.h>
#if EDGELLM_IPC_HAS_NVBUFSURFACE
#include <nvbufsurface.h>
#endif

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace edgellm::ipc
{

//! Frame size, which must be given: a raw NV12 file has no header.
struct Resolution
{
    uint32_t width;
    uint32_t height;
};

//! NV12 chroma rounds up: 7 rows have 4, not 3.
inline uint32_t chromaRows(uint32_t height) noexcept
{
    return (height + 1U) / 2U;
}

//! One packed NV12 frame: luma rows, then chroma rows.
inline size_t nv12Bytes(Resolution resolution) noexcept
{
    return size_t{resolution.width} * resolution.height
        + size_t{2U * ((resolution.width + 1U) / 2U)} * chromaRows(resolution.height);
}

//! @p count elements of T, or as many whole ones as the file holds when 0.
template <typename T>
std::vector<T> readFile(std::string const& path, size_t count = 0)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    size_t const available = file ? static_cast<size_t>(file.tellg()) / sizeof(T) : 0U;
    std::vector<T> data(count == 0 ? available : count);
    file.seekg(0);
    if (!file || data.size() > available
        || !file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T))))
    {
        throw std::runtime_error(
            "cannot read " + std::to_string(data.size() * sizeof(T)) + " bytes from \"" + path + "\"");
    }
    return data;
}

//! Copy @p samples of PCM into the shared memory, which is sized to exactly @p samples.
inline void copyPcmToShm(float* shm, float const* pcm, size_t samples)
{
    if (samples == 0U)
    {
        return;
    }
    std::memcpy(shm, pcm, samples * sizeof(float));
}

// --- NvSciBuf ----------------------------------------------------------------------------------

//! Our writer's attributes reconciled with the service's, and one NV12 buffer allocated from the result.
inline NvSciBufObj allocateNvSciBuf(NvSciBufNeeds required, Resolution resolution, bool blockLinear)
{
    // Even dimensions: an odd frame is padded out to them.
    auto const width = static_cast<int32_t>(resolution.width + resolution.width % 2U);
    auto const height = static_cast<int32_t>(resolution.height + resolution.height % 2U);
    NvSciBufType const bufferType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm const permissions = NvSciBufAccessPerm_ReadWrite;
    bool const needCpuAccess = true; // What NvSciBufObjPutPixels writes through.
    NvSciBufAttrValImageScanType const scanType = NvSciBufScan_ProgressiveType;
    NvSciBufAttrValImageLayoutType const layout
        = blockLinear ? NvSciBufImage_BlockLinearType : NvSciBufImage_PitchLinearType;
    int32_t const planeCount = 2;
    // V8U8 is NV12, NvSciBuf naming formats MSB-first; NvSciColor_U8V8 is NV21 and gets refused.
    NvSciBufAttrValColorFmt const colorFormats[2] = {NvSciColor_Y8, NvSciColor_V8U8};
    // Studio range, the 16-235 luma raw NV12 carries.
    NvSciBufAttrValColorStd const colorStandards[2] = {NvSciColorStd_REC601_SR, NvSciColorStd_REC601_SR};
    int32_t const widths[2] = {width, width / 2};
    int32_t const heights[2] = {height, height / 2};
    NvSciBufAttrKeyValuePair pairs[] = {
        {NvSciBufGeneralAttrKey_Types, &bufferType, sizeof(bufferType)},
        {NvSciBufGeneralAttrKey_RequiredPerm, &permissions, sizeof(permissions)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccess, sizeof(needCpuAccess)},
        {NvSciBufImageAttrKey_Layout, &layout, sizeof(layout)},
        {NvSciBufImageAttrKey_ScanType, &scanType, sizeof(scanType)},
        {NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount)},
        {NvSciBufImageAttrKey_PlaneColorFormat, colorFormats, sizeof(colorFormats)},
        {NvSciBufImageAttrKey_PlaneColorStd, colorStandards, sizeof(colorStandards)},
        {NvSciBufImageAttrKey_PlaneWidth, widths, sizeof(widths)},
        {NvSciBufImageAttrKey_PlaneHeight, heights, sizeof(heights)},
    };
    NvSciBufAttrList writer = nullptr;
    checkNvSci(NvSciBufAttrListCreate(required.module, &writer), "NvSciBufAttrListCreate");
    checkNvSci(NvSciBufAttrListSetAttrs(writer, pairs, sizeof(pairs) / sizeof(pairs[0])), "NvSciBufAttrListSetAttrs");

    // The service's list too, without which the buffer cannot travel.
    NvSciBufAttrList const inputs[] = {writer, required.serviceNeeds};
    NvSciBufAttrList reconciled = nullptr;
    NvSciBufAttrList conflicts = nullptr;
    NvSciError const reconciliation = NvSciBufAttrListReconcile(inputs, 2, &reconciled, &conflicts);
    NvSciBufAttrListFree(writer);
    if (conflicts != nullptr)
    {
        NvSciBufAttrListFree(conflicts);
    }
    checkNvSci(reconciliation, "NvSciBufAttrListReconcile against the service's attributes");

    NvSciBufObj buffer = nullptr;
    NvSciError const allocation = NvSciBufObjAlloc(reconciled, &buffer);
    NvSciBufAttrListFree(reconciled);
    checkNvSci(allocation, "NvSciBufObjAlloc");
    return buffer;
}

//! Copy the NV12 image at @p pixels into the NvSciBufObj.
inline void copyNV12ToNvBuf(NvSciBufObj buffer, Resolution resolution, uint8_t const* pixels)
{
    // PutPixels takes YUV as separate Y, U and V planes and interleaves U and V itself; an interleaved one is refused.
    uint32_t const width = resolution.width + resolution.width % 2U;
    uint32_t const height = resolution.height + resolution.height % 2U;
    std::vector<uint8_t> luma(size_t{width} * height);
    for (uint32_t row = 0; row < resolution.height; ++row)
    {
        std::memcpy(luma.data() + size_t{row} * width, pixels + size_t{row} * resolution.width, resolution.width);
    }
    uint8_t const* const chroma = pixels + size_t{resolution.width} * resolution.height;
    std::vector<uint8_t> u(luma.size() / 4U);
    std::vector<uint8_t> v(u.size());
    for (size_t i = 0; i < u.size(); ++i)
    {
        u[i] = chroma[2U * i];
        v[i] = chroma[2U * i + 1U];
    }
    void const* planes[3] = {luma.data(), u.data(), v.data()};
    uint32_t const sizes[3]
        = {static_cast<uint32_t>(luma.size()), static_cast<uint32_t>(u.size()), static_cast<uint32_t>(v.size())};
    uint32_t const pitches[3] = {width, width / 2U, width / 2U};
    // NULL rect: the whole surface, the only thing PutPixels takes for YUV.
    checkNvSci(NvSciBufObjPutPixels(buffer, nullptr, planes, sizes, pitches), "NvSciBufObjPutPixels");

    // A cached CPU mapping has to be flushed before the service's GPU reads it.
    NvSciBufAttrList list = nullptr;
    NvSciBufAttrKeyValuePair query[]
        = {{NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, nullptr, 0}, {NvSciBufImageAttrKey_Size, nullptr, 0}};
    checkNvSci(NvSciBufObjGetAttrList(buffer, &list), "NvSciBufObjGetAttrList");
    checkNvSci(NvSciBufAttrListGetAttrs(list, query, 2), "NvSciBufAttrListGetAttrs");
    if (*static_cast<bool const*>(query[0].value))
    {
        checkNvSci(NvSciBufObjFlushCpuCacheRange(buffer, 0, *static_cast<uint64_t const*>(query[1].value)),
            "NvSciBufObjFlushCpuCacheRange");
    }
}

inline void destroyNvBuf(NvSciBufObj buffer)
{
    NvSciBufObjFree(buffer);
}

// --- NvBufSurface ------------------------------------------------------------------------------

#if EDGELLM_IPC_HAS_NVBUFSURFACE
//! One NV12 surface out of the multimedia allocator, whose dmabuf is what crosses.
inline NvBufSurface* allocateNvBufSurface(Resolution resolution, bool blockLinear)
{
    // NVBUF_MEM_SURFACE_ARRAY is the only type NvBufSurfaceMapCudaBuffer maps; the tag picks the allocator.
    NvBufSurfaceAllocateParams request{};
    request.params.width = resolution.width + resolution.width % 2U;
    request.params.height = resolution.height + resolution.height % 2U;
    request.params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
    request.params.layout = blockLinear ? NVBUF_LAYOUT_BLOCK_LINEAR : NVBUF_LAYOUT_PITCH;
    request.params.memType = NVBUF_MEM_SURFACE_ARRAY;
    request.displayscanformat = NVBUF_DISPLAYSCANFORMAT_PROGRESSIVE;
    request.memtag = NvBufSurfaceTag_VIDEO_DEC;
    NvBufSurface* surface = nullptr;
    if (NvBufSurfaceAllocate(&surface, 1, &request) != 0)
    {
        throw std::runtime_error("NvBufSurfaceAllocate refused an NV12 surface");
    }
    surface->numFilled = 1;
    // Zeroed for the padding an odd frame leaves, which no fill writes over.
    NvBufSurfaceMemSet(surface, -1, -1, 0);
    return surface;
}

inline void copyNV12ToNvBuf(NvBufSurface* surface, Resolution resolution, uint8_t const* pixels)
{
    NvBufSurfacePlaneParams const& planes = surface->surfaceList[0].planeParams;
    uint32_t const rows[2] = {resolution.height, chromaRows(resolution.height)};
    uint8_t const* const sources[2] = {pixels, pixels + size_t{resolution.width} * resolution.height};
    for (uint32_t i = 0; i < 2U; ++i)
    {
        // Copy this plane with Raw2NvBufSurface, since the surface's row layout is not packed NV12; width is in
        // samples.
        if (Raw2NvBufSurface(const_cast<uint8_t*>(sources[i]), 0, i, planes.width[i], rows[i], surface) != 0)
        {
            throw std::runtime_error("Raw2NvBufSurface refused plane " + std::to_string(i));
        }
    }
}

inline void destroyNvBuf(NvBufSurface* surface)
{
    NvBufSurfaceDestroy(surface);
}
#endif

} // namespace edgellm::ipc
