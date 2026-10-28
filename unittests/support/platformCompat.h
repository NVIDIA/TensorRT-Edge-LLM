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

#pragma once

#include <cstdlib>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace trt_edgellm
{
namespace test
{

inline int processId() noexcept
{
#if defined(_WIN32)
    return _getpid();
#else
    return getpid();
#endif
}

inline void setEnv(char const* name, char const* value) noexcept
{
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

//! Windows removes a variable by assigning it an empty value.
inline void unsetEnv(char const* name) noexcept
{
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

} // namespace test
} // namespace trt_edgellm
