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

#ifdef ENABLE_NVTX_PROFILING
#include <nvtx3.hpp>

//! Macro wrapper for NVTX scoped range - creates RAII profiling marker
//! Usage: NVTX_SCOPED_RANGE(var_name, "range_name", NVTX_RGB(r, g, b))
#define NVTX_SCOPED_RANGE(var_name, ...)                                                                               \
    nvtx3::scoped_range var_name                                                                                       \
    {                                                                                                                  \
        __VA_ARGS__                                                                                                    \
    }

//! Macro wrapper for NVTX color specification
//! Usage: NVTX_RGB(255, 128, 0) for orange color
#define NVTX_RGB(r, g, b)                                                                                              \
    nvtx3::rgb                                                                                                         \
    {                                                                                                                  \
        r, g, b                                                                                                        \
    }

//! Predefined NVTX colors for consistent profiling visualization
namespace nvtx_colors
{
// Warm colors
constexpr auto ORANGE = NVTX_RGB(255, 128, 64);
constexpr auto LIGHT_ORANGE = NVTX_RGB(255, 165, 0);
constexpr auto PALE_ORANGE = NVTX_RGB(255, 200, 100);
constexpr auto DARK_ORANGE = NVTX_RGB(255, 128, 0);
constexpr auto YELLOW = NVTX_RGB(255, 200, 0);

// Cool colors
constexpr auto BLUE = NVTX_RGB(0, 128, 255);
constexpr auto LIGHT_BLUE = NVTX_RGB(64, 192, 255);
constexpr auto SKY_BLUE = NVTX_RGB(100, 200, 255);
constexpr auto GREEN = NVTX_RGB(0, 255, 128);
constexpr auto LIGHT_GREEN = NVTX_RGB(128, 255, 128);
constexpr auto PALE_GREEN = NVTX_RGB(150, 255, 150);

// Pink/Purple tones
constexpr auto PINK = NVTX_RGB(255, 150, 200);
constexpr auto MAGENTA = NVTX_RGB(255, 0, 128);
constexpr auto PURPLE = NVTX_RGB(128, 0, 128);

} // namespace nvtx_colors

#else
//! No-op implementation when NVTX profiling is disabled
//! Compiles to nothing - zero overhead, no namespace pollution
#define NVTX_SCOPED_RANGE(var_name, ...) ((void) 0)
#define NVTX_RGB(r, g, b) 0

#endif
