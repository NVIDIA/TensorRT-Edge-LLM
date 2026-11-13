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

#include "pleFileUtils.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace trt_edgellm
{
namespace rt
{

PleFileHandle openPleFile(std::filesystem::path const& path)
{
#if defined(_WIN32)
    // Share write and delete like a POSIX descriptor: an open shard does not block its file being replaced or removed.
    HANDLE const file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
    if (file == INVALID_HANDLE_VALUE)
    {
        throw std::system_error(GetLastError(), std::system_category(), "Opening PLE shard");
    }
    return reinterpret_cast<PleFileHandle>(file);
#else
    int const file = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (file < 0)
    {
        throw std::system_error(errno, std::generic_category(), "Opening PLE shard");
    }
    return static_cast<PleFileHandle>(file);
#endif
}

void closePleFile(PleFileHandle file) noexcept
{
#if defined(_WIN32)
    CloseHandle(reinterpret_cast<HANDLE>(file));
#else
    ::close(static_cast<int>(file));
#endif
}

size_t readPleFileAt(PleFileHandle file, void* buffer, size_t bytes, int64_t offset, char const* errorMessage)
{
    if (offset < 0)
    {
        throw std::invalid_argument("PLE read offset cannot be negative");
    }
#if defined(_WIN32)
    DWORD const request = static_cast<DWORD>(std::min(bytes, static_cast<size_t>(std::numeric_limits<DWORD>::max())));
    ULARGE_INTEGER position{};
    position.QuadPart = static_cast<ULONGLONG>(offset);
    OVERLAPPED operation{};
    operation.Offset = position.LowPart;
    operation.OffsetHigh = position.HighPart;
    operation.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (operation.hEvent == nullptr)
    {
        throw std::system_error(GetLastError(), std::system_category(), errorMessage);
    }
    DWORD count{};
    BOOL completed = ReadFile(reinterpret_cast<HANDLE>(file), buffer, request, &count, &operation);
    DWORD error = completed ? ERROR_SUCCESS : GetLastError();
    if (!completed && error == ERROR_IO_PENDING)
    {
        completed = GetOverlappedResult(reinterpret_cast<HANDLE>(file), &operation, &count, TRUE);
        error = completed ? ERROR_SUCCESS : GetLastError();
    }
    CloseHandle(operation.hEvent);
    if (!completed)
    {
        if (error == ERROR_HANDLE_EOF)
        {
            return 0;
        }
        throw std::system_error(error, std::system_category(), errorMessage);
    }
    return static_cast<size_t>(count);
#else
    ssize_t count{};
    do
    {
        count = ::pread(static_cast<int>(file), buffer, bytes, static_cast<off_t>(offset));
    } while (count < 0 && errno == EINTR);
    if (count < 0)
    {
        throw std::system_error(errno, std::generic_category(), errorMessage);
    }
    return static_cast<size_t>(count);
#endif
}

size_t pleSystemPageSize() noexcept
{
#if defined(_WIN32)
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return static_cast<size_t>(info.dwPageSize);
#else
    long const result = ::sysconf(_SC_PAGESIZE);
    return result > 0 ? static_cast<size_t>(result) : 0;
#endif
}

} // namespace rt
} // namespace trt_edgellm
