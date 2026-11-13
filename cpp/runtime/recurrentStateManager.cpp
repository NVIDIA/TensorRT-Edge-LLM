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

#include "runtime/recurrentStateManager.h"
#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <limits>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

size_t checkedMultiply(size_t lhs, size_t rhs, std::string const& description)
{
    ELLM_CHECK(rhs == 0 || lhs <= std::numeric_limits<size_t>::max() / rhs,
        "Recurrent state size overflow for " + description);
    return lhs * rhs;
}

//! Per-slot row shape, with the slot dimension prepended.
Coords slabShape(Coords const& slotShape, int32_t slotCount)
{
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(slotShape.getNumDims()) + 1U);
    dims.push_back(slotCount);
    for (int32_t dim = 0; dim < slotShape.getNumDims(); ++dim)
    {
        dims.push_back(slotShape[dim]);
    }
    return Coords{dims};
}

std::string describe(StateSpec const& spec)
{
    return std::string(stateKindName(spec.kind)) + " state on layer " + std::to_string(spec.absLayerIdx);
}

size_t slotBytesOf(StateSpec const& spec)
{
    ELLM_CHECK(spec.shape.getNumDims() > 0, describe(spec) + " has no shape");
    for (int32_t dim = 0; dim < spec.shape.getNumDims(); ++dim)
    {
        ELLM_CHECK(spec.shape[dim] > 0, describe(spec) + " has a non-positive extent " + spec.shape.formatString());
    }
    return checkedMultiply(static_cast<size_t>(spec.shape.volume()), utils::getTypeSize(spec.dataType), describe(spec));
}

std::byte* byteOffset(void* pointer, size_t offset)
{
    return static_cast<std::byte*>(pointer) + offset;
}

std::byte const* byteOffset(void const* pointer, size_t offset)
{
    return static_cast<std::byte const*>(pointer) + offset;
}

bool sameState(StateSpec const& lhs, StateSpec const& rhs)
{
    return lhs.kind == rhs.kind && lhs.absLayerIdx == rhs.absLayerIdx && lhs.shape == rhs.shape
        && lhs.dataType == rhs.dataType;
}

} // namespace

char const* stateKindName(StateKind kind)
{
    switch (kind)
    {
    case StateKind::kGdn: return "gdn";
    case StateKind::kMamba2: return "mamba2";
    case StateKind::kConv: return "conv";
    case StateKind::kNgramConv: return "ngram_conv";
    }
    ELLM_CHECK(false, "Unknown recurrent state kind " + std::to_string(static_cast<int32_t>(kind)));
    return "";
}

size_t RecurrentStateManager::bytesPerSlot(std::vector<StateSpec> const& specs)
{
    size_t bytes{0};
    for (StateSpec const& spec : specs)
    {
        size_t const slotBytes = slotBytesOf(spec);
        ELLM_CHECK(bytes <= std::numeric_limits<size_t>::max() - slotBytes,
            "Recurrent state size overflow for " + describe(spec));
        bytes += slotBytes;
    }
    return bytes;
}

RecurrentStateManager::RecurrentStateManager(
    std::vector<StateSpec> specs, int32_t slotCount, std::string const& name, cudaStream_t stream)
    : mSpecs(std::move(specs))
    , mSlotCount(slotCount)
{
    ELLM_CHECK(slotCount >= 0, "Recurrent state slot count must be non-negative");
    mSlotBytes.reserve(mSpecs.size());
    for (size_t i = 0; i < mSpecs.size(); ++i)
    {
        StateSpec const& spec = mSpecs[i];
        ELLM_CHECK(spec.absLayerIdx >= 0, describe(spec) + " names a negative layer");
        for (size_t earlier = 0; earlier < i; ++earlier)
        {
            ELLM_CHECK(mSpecs[earlier].kind != spec.kind || mSpecs[earlier].absLayerIdx != spec.absLayerIdx,
                "Duplicate " + describe(spec) + "; a kind occurs at most once per layer");
        }
        mSlotBytes.push_back(slotBytesOf(spec));
    }

    if (mSlotCount == 0 || mSpecs.empty())
    {
        return;
    }

    mStates.reserve(mSpecs.size());
    for (size_t i = 0; i < mSpecs.size(); ++i)
    {
        StateSpec const& spec = mSpecs[i];
        mStates.emplace_back(slabShape(spec.shape, mSlotCount), DeviceType::kGPU, spec.dataType,
            name + "::" + stateKindName(spec.kind) + "State_" + std::to_string(spec.absLayerIdx));
        CUDA_CHECK(cudaMemsetAsync(mStates.back().rawPointer(), 0,
            checkedMultiply(mSlotBytes[i], static_cast<size_t>(mSlotCount), describe(spec)), stream));
    }
}

int32_t RecurrentStateManager::slotCount() const noexcept
{
    return mSlotCount;
}

int32_t RecurrentStateManager::numStates() const noexcept
{
    return static_cast<int32_t>(mSpecs.size());
}

StateSpec const& RecurrentStateManager::spec(int32_t stateIdx) const
{
    ELLM_CHECK(stateIdx >= 0 && stateIdx < numStates(),
        "Recurrent state index " + std::to_string(stateIdx) + " is out of range");
    return mSpecs[static_cast<size_t>(stateIdx)];
}

std::vector<StateSpec> const& RecurrentStateManager::specs() const noexcept
{
    return mSpecs;
}

Tensor& RecurrentStateManager::state(int32_t stateIdx)
{
    ELLM_CHECK(stateIdx >= 0 && stateIdx < static_cast<int32_t>(mStates.size()),
        "Recurrent state index " + std::to_string(stateIdx) + " is out of range");
    return mStates[static_cast<size_t>(stateIdx)];
}

Tensor const& RecurrentStateManager::state(int32_t stateIdx) const
{
    ELLM_CHECK(stateIdx >= 0 && stateIdx < static_cast<int32_t>(mStates.size()),
        "Recurrent state index " + std::to_string(stateIdx) + " is out of range");
    return mStates[static_cast<size_t>(stateIdx)];
}

int32_t RecurrentStateManager::findState(StateKind kind, int32_t absLayerIdx) const
{
    for (size_t i = 0; i < mSpecs.size(); ++i)
    {
        if (mSpecs[i].kind == kind && mSpecs[i].absLayerIdx == absLayerIdx)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

size_t RecurrentStateManager::stateBytesPerSlot(int32_t stateIdx) const
{
    ELLM_CHECK(stateIdx >= 0 && stateIdx < numStates(),
        "Recurrent state index " + std::to_string(stateIdx) + " is out of range");
    return mSlotBytes[static_cast<size_t>(stateIdx)];
}

size_t RecurrentStateManager::bytesPerSlot() const
{
    return bytesPerSlot(mSpecs);
}

void RecurrentStateManager::validateSlot(int32_t slot) const
{
    ELLM_CHECK(slot >= 0 && slot < mSlotCount,
        "Recurrent state slot " + std::to_string(slot) + " is outside a " + std::to_string(mSlotCount) + "-slot pool");
}

void RecurrentStateManager::clear(cudaStream_t stream)
{
    for (size_t i = 0; i < mStates.size(); ++i)
    {
        CUDA_CHECK(cudaMemsetAsync(mStates[i].rawPointer(), 0,
            checkedMultiply(mSlotBytes[i], static_cast<size_t>(mSlotCount), describe(mSpecs[i])), stream));
    }
}

void RecurrentStateManager::clearSlot(int32_t slot, cudaStream_t stream)
{
    validateSlot(slot);
    for (size_t i = 0; i < mStates.size(); ++i)
    {
        size_t const bytes = mSlotBytes[i];
        CUDA_CHECK(
            cudaMemsetAsync(byteOffset(mStates[i].rawPointer(), static_cast<size_t>(slot) * bytes), 0, bytes, stream));
    }
}

std::vector<Tensor> RecurrentStateManager::capture(int32_t slot, cudaStream_t stream)
{
    validateSlot(slot);
    std::vector<Tensor> saved;
    saved.reserve(mStates.size());
    for (size_t i = 0; i < mStates.size(); ++i)
    {
        StateSpec const& stateSpec = mSpecs[i];
        size_t const bytes = mSlotBytes[i];
        Tensor checkpoint(slabShape(stateSpec.shape, 1), DeviceType::kGPU, stateSpec.dataType,
            "RecurrentStateManager::captured_" + std::string(stateKindName(stateSpec.kind)) + "_"
                + std::to_string(stateSpec.absLayerIdx));
        CUDA_CHECK(cudaMemcpyAsync(checkpoint.rawPointer(),
            byteOffset(mStates[i].rawPointer(), static_cast<size_t>(slot) * bytes), bytes, cudaMemcpyDeviceToDevice,
            stream));
        saved.push_back(std::move(checkpoint));
    }
    return saved;
}

void RecurrentStateManager::restore(std::vector<Tensor> const& saved, int32_t slot, cudaStream_t stream)
{
    validateSlot(slot);
    ELLM_CHECK(saved.size() <= mStates.size(),
        "Checkpoint holds " + std::to_string(saved.size()) + " states, more than the pool's "
            + std::to_string(mStates.size()));
    for (size_t i = 0; i < saved.size(); ++i)
    {
        StateSpec const& stateSpec = mSpecs[i];
        std::string const description = "Checkpoint of " + describe(stateSpec);
        ELLM_CHECK(saved[i].getShape() == slabShape(stateSpec.shape, 1),
            description + " has shape " + saved[i].getShape().formatString() + ", not "
                + slabShape(stateSpec.shape, 1).formatString());
        ELLM_CHECK(saved[i].getDataType() == stateSpec.dataType, description + " has the wrong element type");
        ELLM_CHECK(saved[i].getDeviceType() == DeviceType::kGPU, description + " must reside on the GPU");
        ELLM_CHECK(
            saved[i].getMemoryCapacity() >= 0 && static_cast<size_t>(saved[i].getMemoryCapacity()) >= mSlotBytes[i],
            description + " backing allocation is smaller than one slot");
    }

    for (size_t i = 0; i < mStates.size(); ++i)
    {
        size_t const bytes = mSlotBytes[i];
        std::byte* destination = byteOffset(mStates[i].rawPointer(), static_cast<size_t>(slot) * bytes);
        if (i < saved.size())
        {
            CUDA_CHECK(cudaMemcpyAsync(destination, saved[i].rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            CUDA_CHECK(cudaMemsetAsync(destination, 0, bytes, stream));
        }
    }
}

void RecurrentStateManager::copySlotFrom(
    RecurrentStateManager const& source, int32_t destinationSlot, int32_t sourceSlot, cudaStream_t stream)
{
    validateSlot(destinationSlot);
    source.validateSlot(sourceSlot);
    ELLM_CHECK(mSpecs.size() == source.mSpecs.size(),
        "Recurrent state pools hold " + std::to_string(mSpecs.size()) + " and " + std::to_string(source.mSpecs.size())
            + " states and cannot exchange a slot");
    for (size_t i = 0; i < mSpecs.size(); ++i)
    {
        ELLM_CHECK(sameState(mSpecs[i], source.mSpecs[i]),
            "Recurrent state pools disagree at position " + std::to_string(i) + ": " + describe(mSpecs[i]) + " versus "
                + describe(source.mSpecs[i]));
    }
    for (size_t i = 0; i < mStates.size(); ++i)
    {
        size_t const bytes = mSlotBytes[i];
        CUDA_CHECK(cudaMemcpyAsync(byteOffset(mStates[i].rawPointer(), static_cast<size_t>(destinationSlot) * bytes),
            byteOffset(source.mStates[i].rawPointer(), static_cast<size_t>(sourceSlot) * bytes), bytes,
            cudaMemcpyDeviceToDevice, stream));
    }
}

} // namespace rt
} // namespace trt_edgellm
