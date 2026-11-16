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

#include "safetyAttentionModules.h"
#include "fmha/safetyFmhaLaunch.h"
#include "safetyFmhaArtifacts.h"
#if defined(EDGELLM_SAFETY_BLACKWELL)
#include "fmha/safetyBlackwellLaunch.h"
#endif

#include <array>
#include <atomic>
#include <iterator>

namespace trt_edgellm
{
namespace safety
{
CUresult SafetyAttentionModules::fail(CUresult status) noexcept
{
    mState = State::kFAILED;
    return status;
}

CUresult SafetyAttentionModules::load() noexcept
{
    static_assert(std::size(fmha::kARTIFACTS) == std::tuple_size_v<decltype(mModules)>);
    if (mState != State::kEMPTY)
    {
        return fail(CUDA_ERROR_INVALID_VALUE);
    }
    mState = State::kLOADING;
    for (size_t i = 0; i < mModules.size(); ++i)
    {
        auto const status = cuModuleLoadData(&mModules[i], fmha::kARTIFACTS[i].cubin);
        if (status != CUDA_SUCCESS)
        {
            return fail(status);
        }
    }
    mState = State::kLOADED;
    return CUDA_SUCCESS;
}

CUresult SafetyAttentionModules::prepare() noexcept
{
    if (mState == State::kREADY)
    {
        return CUDA_SUCCESS;
    }
    if (mState != State::kLOADED)
    {
        return fail(CUDA_ERROR_INVALID_VALUE);
    }
    for (size_t i = 0; i < mModules.size(); ++i)
    {
        auto const& artifact = fmha::kARTIFACTS[i];
        auto status = cuModuleGetFunction(&mFunctions[i], mModules[i], artifact.symbol);
        if (status == CUDA_SUCCESS)
        {
            status = cuFuncSetAttribute(mFunctions[i], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                static_cast<int>(artifact.sharedMemory));
        }
        if (status != CUDA_SUCCESS)
        {
            return fail(status);
        }
    }
#if defined(EDGELLM_SAFETY_BLACKWELL)
    int device{};
    auto status = cudaGetDevice(&device);
    if (status == cudaSuccess)
    {
        status = cudaDeviceGetAttribute(&mSmCount, cudaDevAttrMultiProcessorCount, device);
    }
    if (status != cudaSuccess || mSmCount <= 0)
    {
        return fail(CUDA_ERROR_UNKNOWN);
    }
#endif
    mState = State::kREADY;
    return CUDA_SUCCESS;
}

CUresult SafetyAttentionModules::seal() noexcept
{
    if (mState != State::kREADY)
    {
        return fail(CUDA_ERROR_INVALID_VALUE);
    }
    mState = State::kOPERATIONAL;
    return CUDA_SUCCESS;
}

CUresult SafetyAttentionModules::release() noexcept
{
    if (mState != State::kLOADED && !ready())
    {
        return fail(CUDA_ERROR_INVALID_VALUE);
    }
    mState = State::kCLOSED;
    for (auto module : mModules)
    {
        auto const status = cuModuleUnload(module);
        if (status != CUDA_SUCCESS)
        {
            return fail(status);
        }
    }
    return CUDA_SUCCESS;
}

bool SafetyAttentionModules::ready() const noexcept
{
    return mState == State::kREADY || mState == State::kOPERATIONAL;
}

CUresult SafetyAttentionModules::prepareBindings(fmha::PreparedBindings& bindings, AttentionLaunch const& shape,
    bool visual, half const* queries, half const* keys, half const* values, half* output) noexcept
{
    if (!ready())
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
#if defined(EDGELLM_SAFETY_BLACKWELL)
    std::array<void const*, 4> const pointers{queries, keys, values, output};
    std::array<int32_t, 7> const dimensions{shape.tokens, shape.qHeads, shape.kvHeads, shape.capacity,
        visual ? shape.sequences : 1, visual ? shape.ropeTokens : shape.tokens, shape.headSize};
    if (bindings.ready)
    {
        return bindings.pointers == pointers && bindings.dimensions == dimensions ? CUDA_SUCCESS
                                                                                  : CUDA_ERROR_INVALID_VALUE;
    }
    if (mState != State::kREADY)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (auto const& artifact : fmha::kARTIFACTS)
    {
        if (artifact.headSize == shape.headSize && artifact.visual == visual)
        {
            auto const status = fmha::prepareBlackwellBindings(bindings, artifact, pointers, dimensions);
            return status == CUDA_SUCCESS ? status : fail(status);
        }
    }
    return CUDA_ERROR_INVALID_VALUE;
#else
    (void) bindings;
    (void) shape;
    (void) visual;
    (void) queries;
    (void) keys;
    (void) values;
    (void) output;
    return CUDA_SUCCESS;
#endif
}

CUresult SafetyAttentionModules::launch(fmha::PreparedBindings& bindings, AttentionLaunch const& shape, bool visual,
    half const* queries, half const* keys, half const* values, int32_t const* cuQueryLengths,
    int32_t const* cuKvLengths, int32_t const* sequenceEnds, half* output, int32_t const* valid,
    cudaStream_t stream) noexcept
{
    if (stream == nullptr || prepareBindings(bindings, shape, visual, queries, keys, values, output) != CUDA_SUCCESS)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (size_t i = 0; i < mFunctions.size(); ++i)
    {
        auto const& artifact = fmha::kARTIFACTS[i];
        if (artifact.headSize != shape.headSize || artifact.visual != visual)
        {
            continue;
        }
#if defined(EDGELLM_SAFETY_BLACKWELL)
        auto const status
            = fmha::launchBlackwell(mFunctions[i], artifact, bindings, visual ? sequenceEnds : cuQueryLengths,
                visual ? sequenceEnds : cuKvLengths, valid, shape.scale, mSmCount, stream);
        return status == CUDA_SUCCESS ? status : fail(status);
#else
        (void) bindings;
        if (visual)
        {
            fmha::VisualArguments args{};
            args.validity = valid;
            args.q = queries;
            args.k = keys;
            args.v = values;
            args.output = output;
            args.cuSeqLens = sequenceEnds;
            args.totalTokens = shape.tokens;
            args.qHeads = shape.qHeads;
            args.kvHeads = shape.kvHeads;
            args.sequences = shape.sequences;
            args.maxSeqLength = shape.ropeTokens;
            args.scale = shape.scale;
            auto const status = fmha::launchVisual(mFunctions[i], artifact, args, stream);
            return status == CUDA_SUCCESS ? status : fail(status);
        }
        fmha::ContiguousArguments args{};
        args.validity = valid;
        args.q = queries;
        args.k = keys;
        args.v = values;
        args.output = output;
        args.cuQ = cuQueryLengths;
        args.cuK = cuKvLengths;
        args.queryLength = shape.tokens;
        args.qHeads = shape.qHeads;
        args.kvHeads = shape.kvHeads;
        args.capacity = shape.capacity;
        args.scale = shape.scale;
        auto const status = fmha::launchContiguous(mFunctions[i], artifact, args, stream);
        return status == CUDA_SUCCESS ? status : fail(status);
#endif
    }
    return CUDA_ERROR_INVALID_VALUE;
}
} // namespace safety
} // namespace trt_edgellm
