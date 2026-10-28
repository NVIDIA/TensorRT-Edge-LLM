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

#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{
namespace marlin_repack
{

__host__ __device__ constexpr int32_t packIndex(int32_t index)
{
    return 2 * (index % 4) + index / 4;
}

__host__ __device__ constexpr int32_t outputIndex(int32_t slot)
{
    return 4 * (slot % 32) + slot / 32;
}

__host__ __device__ constexpr int32_t rowIndex(int32_t slot, int32_t packedColumn)
{
    return 2 * (slot % 4) + (packedColumn % 2) + 8 * ((packedColumn / 2) % 2);
}

__host__ __device__ constexpr int32_t columnIndex(int32_t slot, int32_t packedColumn)
{
    int32_t const block = slot / 4;
    return (block / 8) * 16 + block % 8 + (packedColumn < 4 ? 0 : 8);
}

__host__ __device__ constexpr int32_t awqUndoIndex(int32_t index)
{
    return index / 2 + 4 * (index % 2);
}

} // namespace marlin_repack
} // namespace kernel
} // namespace trt_edgellm
