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
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>

namespace trt_edgellm::kernels
{

struct Nvfp4DualGemmParams
{
    void const* activation;        //!< E2M1 bytes [M,K/2], low nibble first.
    void const* packedWeight;      //!< E2M1 [2N,K/2], alternating 128-row gate/up tiles.
    void const* activationScale;   //!< E4M3 SF-atom layout for [M,K/16].
    void const* packedWeightScale; //!< E4M3 SF-atom layout for interleaved [2N,K/16].
    void const* alpha;             //!< Two device FP32 scalars: gate, up; applied before SiLU.
    void* output;                  //!< FP16 [M,N], or E2M1 [M,N/2] when outputScale is set.
    int32_t numTokens;
    int32_t interSize;
    int32_t hiddenSize;
    int32_t maxActiveClusters;
    void* outputScale{};        //!< Optional E4M3 SF atoms for [M,N/16], padded to 128 rows.
    float const* globalScale{}; //!< Device forward scale for quantized output.
    int32_t tactic{};           //!< 0/1: fixed 2x1, 2: mixed 2x2/2x1, 3: mixed 4x1/2x1 (quantized output).
};

class CuteDslNvfp4DualGemmRunner
{
public:
    static constexpr int32_t kDEFAULT_TACTIC = 0;
    static constexpr int32_t kFIXED_2X1_TACTIC = 1;
    static constexpr int32_t kMIXED_2X2_TACTIC = 2;
    static constexpr int32_t kMIXED_4X1_TACTIC = 3;
    static constexpr int32_t kNUM_QUANTIZED_TACTICS = kMIXED_4X1_TACTIC;

    static bool canImplement(int32_t smVersion) noexcept;
    //! Initialize outside capture. Returns the persistent worker count for this device.
    cudaError_t prepare(
        int32_t& maxActiveClusters, bool quantizedOutput = false, int32_t tactic = kDEFAULT_TACTIC) noexcept;
    //! Launch only; a missing prepare() is an initialization error.
    cudaError_t run(Nvfp4DualGemmParams const& params, cudaStream_t stream) const noexcept;

private:
    // Keep profiled tactics alive until this execution instance is destroyed.
    std::array<std::shared_ptr<void>, kNUM_QUANTIZED_TACTICS + 1> mModules;
};

} // namespace trt_edgellm::kernels
