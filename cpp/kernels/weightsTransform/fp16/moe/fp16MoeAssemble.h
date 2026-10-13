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

enum class Fp16MoeSourceType : int32_t
{
    kFp16,
    kBf16,
    kFp32,
};

//! Cast and stack checkpoint matrices directly into an FP16 MoE buffer.
//! When secondSources is non-null, each source pair is emitted in the
//! Fp16MoePlugin's fixed 64-row up/gate interleave.
cudaError_t launchFp16MoeSourceBatch(uint8_t const* const* firstSources, uint8_t const* const* secondSources,
    int32_t count, void* output, int32_t sourceRows, int32_t sourceColumns, int32_t outputRows, int32_t outputColumns,
    Fp16MoeSourceType sourceType, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
