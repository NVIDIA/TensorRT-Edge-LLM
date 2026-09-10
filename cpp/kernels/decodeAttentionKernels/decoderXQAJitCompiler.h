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

#include <NvInferRuntime.h>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

namespace trt_edgellm
{

//! \brief Key that identifies one NVRTC-compiled XQA decode kernel variant.
struct XQAJitKey
{
    int32_t sm{};
    nvinfer1::DataType dataType{nvinfer1::DataType::kHALF};
    nvinfer1::DataType kvDataType{nvinfer1::DataType::kHALF};
    int32_t headSize{};
    int32_t qHeadsPerKv{};
    int32_t tokensPerPage{};
    bool slidingWindow{};
    bool specDecode{};

    auto asTuple() const noexcept
    {
        return std::tie(sm, dataType, kvDataType, headSize, qHeadsPerKv, tokensPerPage, slidingWindow, specDecode);
    }

    bool operator==(XQAJitKey const& other) const noexcept
    {
        return asTuple() == other.asTuple();
    }
};

//! \brief Result of compiling an XQA kernel with NVRTC.
struct XQAJitResult
{
    std::vector<uint8_t> cubin;
};

//! \brief One NVRTC-compiled kernel together with the key it was compiled for.
struct XQAJitKernel
{
    XQAJitKey key;
    std::vector<uint8_t> cubin;
};

//! \brief Serialize compiled XQA kernels into a self-describing byte blob.
//!
//! The key is serialized field by field alongside its cubin so that a
//! deserialized kernel is registered under the key it was actually compiled
//! for, rather than one recomputed from whatever plugin fields happen to be
//! present at deserialization time.
std::vector<uint8_t> serializeXQAJitKernels(std::vector<XQAJitKernel> const& kernels);

//! \brief Inverse of serializeXQAJitKernels.
//! \throws std::runtime_error if the blob is truncated or carries an unknown format version.
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

//! \brief Check if a given configuration can produce a valid XQA JIT kernel.
bool canCompileXQAKernel(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
    nvinfer1::DataType dataType, nvinfer1::DataType kvDataType) noexcept;

//! \brief Compile one XQA kernel variant with NVRTC.
//! \throws std::runtime_error if NVRTC support is disabled or compilation fails.
XQAJitResult compileXQAKernel(XQAJitKey const& key);

} // namespace trt_edgellm
