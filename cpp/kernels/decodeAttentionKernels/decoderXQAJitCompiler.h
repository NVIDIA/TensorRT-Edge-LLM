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

#include "kernels/PluginJitKernels/pluginJitKey.h"
#include "kernels/decodeAttentionKernels/xqaKernelSupport.h"

#include "common/dataType.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trt_edgellm
{

//! \brief Key that identifies one NVRTC-compiled XQA decode kernel variant.
//! Follows the plugin JIT key contract in pluginJitKey.h.
struct XQAJitKey
{
    int32_t sm{};
    trt_edgellm::DataType dataType{trt_edgellm::DataType::kHALF};
    trt_edgellm::DataType kvDataType{trt_edgellm::DataType::kHALF};
    int32_t headSize{};
    int32_t qHeadsPerKv{};
    int32_t tokensPerPage{};
    bool slidingWindow{};
    bool specDecode{};
    bool contiguousQuerySwa{};

    static constexpr size_t kNumWords{9};

    //! The fields in blob, digest and hash order.
    std::array<uint32_t, kNumWords> keyWords() const noexcept
    {
        return {static_cast<uint32_t>(sm), static_cast<uint32_t>(dataType), static_cast<uint32_t>(kvDataType),
            static_cast<uint32_t>(headSize), static_cast<uint32_t>(qHeadsPerKv), static_cast<uint32_t>(tokensPerPage),
            slidingWindow ? 1U : 0U, specDecode ? 1U : 0U, contiguousQuerySwa ? 1U : 0U};
    }

    static XQAJitKey fromKeyWords(std::array<uint32_t, kNumWords> const& words) noexcept
    {
        XQAJitKey key;
        key.sm = static_cast<int32_t>(words[0]);
        key.dataType = static_cast<trt_edgellm::DataType>(words[1]);
        key.kvDataType = static_cast<trt_edgellm::DataType>(words[2]);
        key.headSize = static_cast<int32_t>(words[3]);
        key.qHeadsPerKv = static_cast<int32_t>(words[4]);
        key.tokensPerPage = static_cast<int32_t>(words[5]);
        key.slidingWindow = words[6] != 0U;
        key.specDecode = words[7] != 0U;
        key.contiguousQuerySwa = words[8] != 0U;
        return key;
    }

    bool operator==(XQAJitKey const& other) const noexcept
    {
        return keyWords() == other.keyWords();
    }
};

//! \brief One NVRTC-compiled kernel together with the key it was compiled for and the digest binding the two.
using XQAJitKernel = PluginJitKernel<XQAJitKey>;

//! \brief Serialize compiled XQA kernels into a self-describing byte blob.
//!
//! The key is serialized field by field alongside its cubin so that a
//! deserialized kernel is registered under the key it was actually compiled
//! for, rather than one recomputed from whatever plugin fields happen to be
//! present at deserialization time.
std::vector<uint8_t> serializeXQAJitKernels(std::vector<XQAJitKernel> const& kernels);

//! \brief Inverse of serializeXQAJitKernels.
//! \throws std::runtime_error if the blob is truncated, tampered with, or does not open with the XQA magic word.
std::vector<XQAJitKernel> deserializeXQAJitKernels(void const* data, size_t size);

//! \brief Return the M tile size used by both NVRTC compilation and host launch metadata.
inline int32_t getXQAJitMTileSize(XQAJitKey const& key) noexcept
{
    constexpr int32_t kDECODE_M_TILE_SIZE{8};
    constexpr int32_t kSPEC_DECODE_M_TILE_SIZE{32};
    constexpr int32_t kSM80_SPEC_DECODE_HEAD_DIM512_M_TILE_SIZE{16};
    constexpr int32_t kHEAD_DIM_512{512};
    constexpr int32_t kSM80{80};
    constexpr int32_t kSM87{87};

    if (key.specDecode && (key.sm == kSM80 || key.sm == kSM87) && key.headSize == kHEAD_DIM_512)
    {
        return kSM80_SPEC_DECODE_HEAD_DIM512_M_TILE_SIZE;
    }
    return key.specDecode ? kSPEC_DECODE_M_TILE_SIZE : kDECODE_M_TILE_SIZE;
}

//! \brief Compile one XQA kernel variant with NVRTC.
//! \throws std::runtime_error if NVRTC support is disabled or compilation fails.
XQAJitKernel compileXQAKernel(XQAJitKey const& key);

} // namespace trt_edgellm
