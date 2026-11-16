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

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace trt_edgellm
{
namespace rt
{

using PleFileHandle = std::intptr_t;

//! Open a PLE shard for positioned reads.
PleFileHandle openPleFile(std::filesystem::path const& path);

//! Close a PLE shard handle.
void closePleFile(PleFileHandle file) noexcept;

//! Read at most bytes from a fixed file offset, returning zero at end of file.
size_t readPleFileAt(PleFileHandle file, void* buffer, size_t bytes, int64_t offset, char const* errorMessage);

//! Return the host memory page size, or zero when the platform query fails.
size_t pleSystemPageSize() noexcept;

} // namespace rt
} // namespace trt_edgellm
