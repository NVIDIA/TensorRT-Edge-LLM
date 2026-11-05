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
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

cudaError_t launchModelOptInt4PluginV1Repack(
    uint8_t const* weightNhalfK, int8_t* packedNhalfK, int32_t N, int32_t K, cudaStream_t stream);

cudaError_t launchGptqInt4PluginV1Repack(int32_t const* qweightK8N, int32_t const* qzerosGN8,
    int32_t const* activationPermutation, int8_t* packedNhalfK, int32_t N, int32_t K, int32_t numGroups,
    int32_t groupSize, int32_t zeroPointOffset, cudaStream_t stream);

cudaError_t launchAwqInt4PluginV1Repack(int32_t const* qweightKN8, int32_t const* qzerosGN8, int8_t* packedNhalfK,
    int32_t N, int32_t K, int32_t numGroups, int32_t groupSize, cudaStream_t stream);

cudaError_t launchAwqCenteredInt4PluginV1Repack(
    int32_t const* qweightKN8, int8_t* packedNhalfK, int32_t N, int32_t K, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
