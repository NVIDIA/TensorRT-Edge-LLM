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

#include <array>
#include <cstddef>
#include <cstdint>

namespace trt_edgellm
{

//! True when the payload starts with the ELF magic, i.e. is a native cubin rather than PTX or
//! garbage. Plugin JIT compiles for the real device architecture and only ever hands native
//! modules to cuModuleLoadData; a virtual-architecture PTX image is not part of the contract.
inline bool isNativeCubin(void const* const data, size_t const size) noexcept
{
    constexpr std::array<uint8_t, 4> kELF_MAGIC{0x7FU, 0x45U, 0x4CU, 0x46U};
    if (data == nullptr || size < kELF_MAGIC.size())
    {
        return false;
    }
    auto const* const bytes = static_cast<uint8_t const*>(data);
    return bytes[0] == kELF_MAGIC[0] && bytes[1] == kELF_MAGIC[1] && bytes[2] == kELF_MAGIC[2]
        && bytes[3] == kELF_MAGIC[3];
}

} // namespace trt_edgellm
