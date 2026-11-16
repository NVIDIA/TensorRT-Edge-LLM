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

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Number of INT8 ``[rows, 512]`` rows required by Int4GroupwiseGemmPluginV2.
int32_t int4CuteDslFragmentRows(int32_t N, int32_t K);

//! Repack ModelOpt packed ``[N/2,K]`` weights directly from checkpoint storage.
cudaError_t launchModelOptInt4CuteDslRepack(
    uint8_t const* weightNhalfK, int8_t* fragmentRows512, int32_t N, int32_t K, cudaStream_t stream);

//! Repack GPTQ ``[K/8,N]`` weights directly from checkpoint storage.
cudaError_t launchGptqInt4CuteDslRepack(int32_t const* qweightK8N, int32_t const* qzerosGN8,
    int32_t const* activationPermutation, int8_t* fragmentRows512, int32_t N, int32_t K, int32_t numGroups,
    int32_t groupSize, int32_t zeroPointOffset, bool outputMajor, cudaStream_t stream);

//! Repack classic AWQ ``[K,N/8]`` weights directly from checkpoint storage.
cudaError_t launchAwqInt4CuteDslRepack(
    int32_t const* qweightKN8, int8_t* fragmentRows512, int32_t N, int32_t K, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
