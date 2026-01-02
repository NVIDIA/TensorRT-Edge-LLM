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

#include "stringUtils.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace trt_edgellm
{
namespace format
{
namespace
{
/**
 * @brief Format a string using va_list arguments
 * @param fmt Format string
 * @param args Variable arguments list
 * @return Formatted string
 */
std::string vformat(char const* fmt, va_list args)
{
    va_list args0;
    va_copy(args0, args);
    auto const size = vsnprintf(nullptr, 0, fmt, args0);
    if (size <= 0)
    {
        return "";
    }

    // Ensure that the underlying string buffer is large enough, even for
    // the null terminator
    std::string stringBuf(size + 1, char{});
    auto const size2 = std::vsnprintf(&stringBuf[0], size + 1, fmt, args);

    if (size2 != size)
    {
        throw std::runtime_error(std::string(std::strerror(errno)));
    }
    stringBuf.resize(size);

    return stringBuf;
}
} // anonymous namespace
std::string fmtstr(char const* format, ...)
{
    va_list args;
    va_start(args, format);
    std::string result = vformat(format, args);
    va_end(args);
    return result;
}

} // namespace format
} // namespace trt_edgellm
