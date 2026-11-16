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
#include <cuda_runtime_api.h>

namespace trt_edgellm
{
namespace kernel
{

cudaError_t launchNvfp4W4A16MarlinRepackSourceBatch(uint8_t const* const* firstWeights,
    uint8_t const* const* secondWeights, int32_t* marlinOutput, int32_t count, int32_t sourceN, int32_t sourceK,
    int32_t outputN, int32_t outputK, cudaStream_t stream);

cudaError_t launchNvfp4W4A16ScaleRepack(uint8_t const* const* firstSources, uint8_t const* const* secondSources,
    int8_t* marlinOutput, int32_t count, int32_t sourceN, int32_t sourceGroups, int32_t outputN, int32_t outputGroups,
    cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
