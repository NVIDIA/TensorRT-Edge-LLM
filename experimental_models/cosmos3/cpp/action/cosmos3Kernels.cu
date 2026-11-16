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

#include "action/cosmos3Kernels.h"
#include "common/checkMacros.h"
#include "common/counterRng.h"
#include "common/cudaUtils.h"

namespace trt_edgellm
{
namespace cosmos3
{
namespace kernel
{

namespace
{
constexpr int32_t kBlockSize = 256;

__global__ void fusedCombineKernel(float* out, int64_t numElems, float c0, float const* v0, float c1, float const* v1,
    float c2, float const* v2, float c3, float const* v3)
{
    int64_t const i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numElems)
    {
        return;
    }
    float acc = c0 * v0[i] + c1 * v1[i];
    if (v2 != nullptr)
    {
        acc += c2 * v2[i];
    }
    if (v3 != nullptr)
    {
        acc += c3 * v3[i];
    }
    out[i] = acc;
}

__global__ void normalNoiseFillKernel(float* out, int64_t numElems, uint64_t seed, uint64_t offset)
{
    int64_t const i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numElems)
    {
        return;
    }
    // Counter-based hash keyed on the element index: reproducible for a given (seed, offset)
    // independent of the launch configuration.
    out[i] = counter_rng::normalFromBits(counter_rng::counterBits(seed, offset, static_cast<uint64_t>(i)));
}

__global__ void buildMRopePositionsKernel(float* positions, int32_t batch, int32_t genLen, int32_t numVideoTokens,
    int32_t hp, int32_t wp, int32_t actionStartFrameOffset, int32_t videoFpsMod, double mediaOffset, double videoTps,
    double videoBaseTps, double actionTps, double actionBaseTps)
{
    int32_t const idx = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= batch * genLen)
    {
        return;
    }
    int32_t const b = idx / genLen;
    int32_t const tok = idx - b * genLen; // gen-sequence token index for this batch element
    double tPos;
    double hPos;
    double wPos;
    if (tok < numVideoTokens)
    {
        int32_t const hw = hp * wp;
        int32_t const ti = tok / hw;
        int32_t const rem = tok - ti * hw;
        int32_t const hi = rem / wp;
        tPos = videoFpsMod ? (static_cast<double>(ti) / videoTps * videoBaseTps + mediaOffset)
                           : (static_cast<double>(ti) + mediaOffset);
        hPos = static_cast<double>(hi);
        wPos = static_cast<double>(rem - hi * wp);
    }
    else
    {
        int32_t const ai = tok - numVideoTokens;
        tPos = static_cast<double>(ai + actionStartFrameOffset) / actionTps * actionBaseTps + mediaOffset;
        hPos = 0.0;
        wPos = 0.0;
    }
    // [B, 3, genLen] plane layout: T at +0, H at +genLen, W at +2*genLen.
    float* base = positions + static_cast<int64_t>(b) * 3 * genLen + tok;
    base[0] = static_cast<float>(tPos);
    base[genLen] = static_cast<float>(hPos);
    base[2 * genLen] = static_cast<float>(wPos);
}
} // namespace

void launchFusedCombine(rt::Tensor& out, float c0, rt::Tensor const& v0, float c1, rt::Tensor const& v1, float c2,
    rt::Tensor const* v2, float c3, rt::Tensor const* v3, cudaStream_t stream)
{
    int64_t const numElems = out.getShape().volume();
    ELLM_CHECK(v0.getShape().volume() == numElems && v1.getShape().volume() == numElems
            && (v2 == nullptr || v2->getShape().volume() == numElems)
            && (v3 == nullptr || v3->getShape().volume() == numElems),
        "launchFusedCombine operands must have identical element counts");
    int32_t const grid = static_cast<int32_t>(divUp(numElems, kBlockSize));
    fusedCombineKernel<<<grid, kBlockSize, 0, stream>>>(static_cast<float*>(out.rawPointer()), numElems, c0,
        static_cast<float const*>(v0.rawPointer()), c1, static_cast<float const*>(v1.rawPointer()), c2,
        v2 != nullptr ? static_cast<float const*>(v2->rawPointer()) : nullptr, c3,
        v3 != nullptr ? static_cast<float const*>(v3->rawPointer()) : nullptr);
}

void launchNormalNoiseFill(rt::Tensor& out, uint64_t seed, uint64_t offset, cudaStream_t stream)
{
    int64_t const numElems = out.getShape().volume();
    int32_t const grid = static_cast<int32_t>(divUp(numElems, kBlockSize));
    normalNoiseFillKernel<<<grid, kBlockSize, 0, stream>>>(
        static_cast<float*>(out.rawPointer()), numElems, seed, offset);
}

void launchBuildMRopePositions(rt::Tensor& positions, int32_t batch, int32_t genLen, int32_t numVideoTokens, int32_t hp,
    int32_t wp, int32_t actionStartFrameOffset, bool videoFpsMod, double mediaOffset, double videoTps,
    double videoBaseTps, double actionTps, double actionBaseTps, cudaStream_t stream)
{
    int32_t const grid = static_cast<int32_t>(divUp(static_cast<int64_t>(batch) * genLen, kBlockSize));
    buildMRopePositionsKernel<<<grid, kBlockSize, 0, stream>>>(static_cast<float*>(positions.rawPointer()), batch,
        genLen, numVideoTokens, hp, wp, actionStartFrameOffset, static_cast<int32_t>(videoFpsMod), mediaOffset,
        videoTps, videoBaseTps, actionTps, actionBaseTps);
}

} // namespace kernel
} // namespace cosmos3
} // namespace trt_edgellm
