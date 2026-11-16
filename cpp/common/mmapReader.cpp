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

#include "mmapReader.h"

#include "common/checkMacros.h"
#include "stringUtils.h"
#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace trt_edgellm
{

namespace file_io
{

MmapReader::MmapReader() noexcept
    : mData(nullptr)
    , mBytes(0)
{
}

MmapReader::MmapReader(std::filesystem::path const& fp, Mode mode)
    : mData(nullptr)
    , mBytes(0)
{
    bool const fileLoaded = loadFile(fp, mode);
    ELLM_CHECK(fileLoaded, "Failed to load file in MmapReader constructor");
}

MmapReader::~MmapReader() noexcept
{
    release();
}

void MmapReader::release() noexcept
{
#if defined(_WIN32)
    if (mData != nullptr)
    {
        UnmapViewOfFile(mData);
    }
#else
    if (mData != nullptr && mBytes > 0)
    {
        munmap(mData, mBytes);
    }
#endif
    mData = nullptr;
    mBytes = 0;
}

bool MmapReader::loadFile(std::filesystem::path const& fp, Mode mode)
{
    // Release any existing memory
    release();

    std::string const filePath = fp.string();
#if defined(_WIN32)
    HANDLE const file = CreateFileW(
        fp.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
    {
        std::cerr << format::fmtstr("MmapReader: Cannot open file: %s", filePath.c_str()) << std::endl;
        return false;
    }
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0)
    {
        CloseHandle(file);
        std::cerr << format::fmtstr("MmapReader: File %s is empty or unreadable.", filePath.c_str()) << std::endl;
        return false;
    }
    mBytes = static_cast<size_t>(size.QuadPart);

    DWORD const protection = mode == Mode::kCopyOnWrite ? PAGE_WRITECOPY : PAGE_READONLY;
    HANDLE const mapping = CreateFileMappingW(file, nullptr, protection, 0, 0, nullptr);
    if (mapping == nullptr)
    {
        CloseHandle(file);
        mBytes = 0;
        std::cerr << format::fmtstr("MmapReader: CreateFileMapping failed for file: %s", filePath.c_str()) << std::endl;
        return false;
    }
    DWORD const access = mode == Mode::kCopyOnWrite ? FILE_MAP_COPY : FILE_MAP_READ;
    mData = MapViewOfFile(mapping, access, 0, 0, 0);
    CloseHandle(mapping);
    CloseHandle(file);
    if (mData == nullptr)
    {
        mBytes = 0;
        std::cerr << format::fmtstr("MmapReader: MapViewOfFile failed for file: %s", filePath.c_str()) << std::endl;
        return false;
    }
#else
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd < 0)
    {
        std::string errorMsg = format::fmtstr("MmapReader: Cannot open file: %s", filePath.c_str());
        std::cerr << errorMsg << std::endl;
        return false;
    }

    struct stat status;
    if (fstat(fd, &status) != 0)
    {
        close(fd);
        std::string errorMsg = format::fmtstr("MmapReader: fstat failed for file: %s", filePath.c_str());
        std::cerr << errorMsg << std::endl;
        return false;
    }
    mBytes = status.st_size;
    if (mBytes == 0)
    {
        close(fd);
        std::string errorMsg = format::fmtstr("MmapReader: File %s is empty.", filePath.c_str());
        std::cerr << errorMsg << std::endl;
        return false;
    }
    int const protection = mode == Mode::kCopyOnWrite ? PROT_READ | PROT_WRITE : PROT_READ;
    int const flags = mode == Mode::kCopyOnWrite ? MAP_PRIVATE : MAP_SHARED;
    mData = mmap(nullptr, mBytes, protection, flags, fd, 0);
    if (mData == MAP_FAILED)
    {
        mData = nullptr;
        mBytes = 0;
        close(fd);
        std::string errorMsg = format::fmtstr("MmapReader: mmap failed for file: %s", filePath.c_str());
        std::cerr << errorMsg << std::endl;
        return false;
    }
    close(fd);
#endif
    return true;
}

int8_t const* MmapReader::getByteData() const noexcept
{
    return reinterpret_cast<int8_t const*>(mData);
}

void const* MmapReader::getData() const noexcept
{
    return mData;
}

size_t MmapReader::getSize() const noexcept
{
    return mBytes;
}

} // namespace file_io
} // namespace trt_edgellm
