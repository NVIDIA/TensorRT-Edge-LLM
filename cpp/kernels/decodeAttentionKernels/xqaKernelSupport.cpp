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

#include "xqaKernelSupport.h"

#include <algorithm>
#include <array>

namespace trt_edgellm
{

bool canCompileXQAKernel(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
    trt_edgellm::DataType dataType, trt_edgellm::DataType kvDataType) noexcept
{
    // SM versions with an XQA decode kernel at all.
    constexpr std::array<int32_t, 10> kALLOWED_SM_VERSIONS{80, 86, 87, 89, 90, 100, 101, 103, 120, 121};
    auto const contains = [](auto const& versions, int32_t sm) noexcept {
        return std::find(versions.begin(), versions.end(), sm) != versions.end();
    };

    bool const checkHeadNumbers = numQHeads % numKVHeads == 0;
    bool const checkType = dataType == trt_edgellm::DataType::kHALF;
    bool const checkKVType = kvDataType == trt_edgellm::DataType::kHALF || kvDataType == trt_edgellm::DataType::kFP8;
    bool const checkSMVersion = contains(kALLOWED_SM_VERSIONS, smVersion);

    // Current kernel list supports
    // (1) Head ratio 1-8 for head_dim {32, 64, 128}
    // (2) Head ratio 16 for head_dim 128 only (NemotronH).
    // (3) Head ratio 2, 4, 6, 8, 16 for head_dim 256
    //     (4/6/8 for Qwen3.5-MoE / Qwen3.5-Omni Thinker+Talker;
    //      2 for Qwen3.5-Omni Talker decode attention — 16 Q heads / 8 KV heads;
    //      16 for Qwen3.5-122B-A10B gated full attention — 32 Q heads / 2 KV heads).
    // (4) Head ratio 2, 4, 8, 16 for head_dim 512.
    //     (2 for Gemma4 E4B assistant: 4 Q heads / 2 KV heads;
    //      4 for Gemma4 E4B: 8 Q heads / 2 KV heads;
    //      8 for Gemma4 E2B: 8 Q heads / 1 KV head;
    //      16 for Gemma4 Unified 12B global attention / single-KV-head decode).
    //     The pre-JIT list additionally gated d512 on which cubins had been
    //     generated for the SM; JIT compiles on demand, so only the ratio
    //     constraint remains.
    int32_t const headRatio = numQHeads / numKVHeads;
    bool const checkQHeadPerKV
        = ((headSize == 32 || headSize == 64 || headSize == 128) && headRatio >= 1 && headRatio <= 8)
        || (headSize == 128 && headRatio == 16)
        || (headSize == 256
            && (headRatio == 2 || headRatio == 4 || headRatio == 6 || headRatio == 8 || headRatio == 16))
        || (headSize == 512 && (headRatio == 2 || headRatio == 4 || headRatio == 8 || headRatio == 16));

    return checkHeadNumbers && checkType && checkKVType && checkSMVersion && checkQHeadPerKV;
}

} // namespace trt_edgellm
