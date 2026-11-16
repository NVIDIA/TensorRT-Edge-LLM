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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace trt_edgellm
{
namespace kernel
{

//! Parameters for packed, segment-routed FP16 LoRA grouped GEMM.
//!
//! The operation is `output += scale[g] * (input * A[g]) * B[g]` for every packed row in segment `g`.
//! Adapter rank zero leaves the corresponding output rows unchanged. `maxRank` is the logical maximum adapter rank;
//! the physical rank stride of the adapter pools is `roundUp(maxRank, 8)`. All metadata and tensor pointers are device
//! pointers. Adapter weights use fixed-stride storage so adapters with different ranks can share one invocation. Weight
//! values outside each adapter's logical rank are ignored and need not be initialized.
struct GroupedLoraGemmParams
{
    half const* input{};                //!< Packed activations, [totalRows, inputSize].
    half const* loraA{};                //!< Adapter A, [numAdapters, inputSize, roundUp(maxRank, 8)].
    half const* loraB{};                //!< Adapter B, [numAdapters, roundUp(maxRank, 8), outputSize].
    int32_t const* segmentOffsets{};    //!< Packed row offsets, [numSegments + 1].
    int32_t const* segmentAdapterIds{}; //!< Adapter slot per segment, [numSegments].
    int32_t const* adapterRanks{};      //!< Effective rank per adapter slot, [numAdapters].
    float const* adapterScales{};       //!< Scale per adapter slot, [numAdapters].
    half* workspace{};                  //!< Caller-owned intermediate and CuTe DSL metadata scratch.
    half* output{};                     //!< Packed accumulation target, [totalRows, outputSize].

    int32_t totalRows{};
    int32_t inputSize{};
    int32_t outputSize{};
    int32_t maxRank{}; //!< Logical maximum adapter rank. The kernel derives the aligned physical stride.
    int32_t numSegments{};
    int32_t numAdapters{};
};

//! Return the physical adapter-pool rank stride for a logical maximum rank, or zero for an invalid rank.
int32_t getGroupedLoraGemmRankStride(int32_t maxRank) noexcept;

//! Return the workspace size for GroupedLoraGemmParams::workspace, or zero for invalid dimensions.
//!
//! `maxRank` is the logical maximum rank; the returned size accounts for its aligned physical stride.
size_t getGroupedLoraGemmWorkspaceSize(int32_t totalRows, int32_t outputSize, int32_t maxRank) noexcept;

//! Load the CuTe DSL module before launch or CUDA graph capture.
//!
//! Returns cudaErrorNotSupported when the build has no compatible artifact.
cudaError_t initializeGroupedLoraGemmCuteDsl(cudaStream_t stream) noexcept;

//! Launch the two grouped GEMMs on an explicit CUDA stream. The second GEMM fuses scale and output accumulation.
//!
//! Metadata must describe monotonically increasing, non-empty segments covering `[0, totalRows)`. Adapter IDs and
//! ranks must be within `numAdapters` and the logical `maxRank`. The adapter pools use `roundUp(maxRank, 8)` as their
//! physical rank stride. Returns cudaErrorNotSupported when the CuTe DSL module is unavailable or the dimensions are
//! unsupported.
cudaError_t launchGroupedLoraGemm(GroupedLoraGemmParams const& params, cudaStream_t stream) noexcept;

} // namespace kernel
} // namespace trt_edgellm
