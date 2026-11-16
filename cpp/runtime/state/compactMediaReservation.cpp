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

#include "runtime/state/compactMediaReservation.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "runtime/llmRuntimeUtils.h"

#include <algorithm>
#include <cuda_fp16.h>
#include <string>

namespace trt_edgellm
{
namespace rt
{

namespace
{

Tensor allocateRegions(int32_t slots, int32_t rowsPerSlot, int32_t hiddenSize, char const* name)
{
    if (rowsPerSlot == 0)
    {
        return Tensor{};
    }
    return Tensor(
        {static_cast<int64_t>(slots) * rowsPerSlot, hiddenSize}, DeviceType::kGPU, trt_edgellm::DataType::kHALF, name);
}

void checkSource(Tensor const& source, int64_t rows, int32_t hiddenSize, std::string const& what)
{
    auto const shape = source.getShape();
    ELLM_CHECK(source.getDeviceType() == DeviceType::kGPU && source.getDataType() == trt_edgellm::DataType::kHALF
            && shape.getNumDims() == 2 && shape[1] == hiddenSize,
        what + " must be a [rows, " + std::to_string(hiddenSize) + "] FP16 device tensor, got " + shape.formatString());
    ELLM_CHECK(shape[0] >= rows,
        what + " holds " + std::to_string(shape[0]) + " rows for the prompt's " + std::to_string(rows) + " positions");
}

} // namespace

CompactMediaReservation::CompactMediaReservation(Shape const& shape)
    : mShape(shape)
    , mSlots(static_cast<size_t>(shape.slots))
{
    ELLM_CHECK(shape.slots > 0 && shape.hiddenSize > 0 && shape.imageRowsPerSlot >= 0 && shape.audioRowsPerSlot >= 0
            && shape.deepstackLayers >= 0 && (shape.deepstackLayers == 0 || shape.imageRowsPerSlot > 0),
        "CompactMediaReservation: invalid shape");
    mImage = allocateRegions(shape.slots, shape.imageRowsPerSlot, shape.hiddenSize, "CompactMediaReservation::image");
    mAudio = allocateRegions(shape.slots, shape.audioRowsPerSlot, shape.hiddenSize, "CompactMediaReservation::audio");
    mDeepstack.reserve(static_cast<size_t>(shape.deepstackLayers));
    for (int32_t layer = 0; layer < shape.deepstackLayers; ++layer)
    {
        mDeepstack.push_back(allocateRegions(
            shape.slots, shape.imageRowsPerSlot, shape.hiddenSize, "CompactMediaReservation::deepstack"));
    }
    mDeepstackRows.assign(mDeepstack.begin(), mDeepstack.end());
}

CompactMediaReservation::SlotMedia& CompactMediaReservation::slotMedia(int32_t slot)
{
    ELLM_CHECK(slot >= 0 && slot < mShape.slots,
        "CompactMediaReservation: slot " + std::to_string(slot) + " is outside [0, " + std::to_string(mShape.slots)
            + ")");
    return mSlots[static_cast<size_t>(slot)];
}

CompactMediaReservation::SlotMedia const& CompactMediaReservation::slotMedia(int32_t slot) const
{
    ELLM_CHECK(slot >= 0 && slot < mShape.slots,
        "CompactMediaReservation: slot " + std::to_string(slot) + " is outside [0, " + std::to_string(mShape.slots)
            + ")");
    return mSlots[static_cast<size_t>(slot)];
}

void CompactMediaReservation::stage(int32_t slot, std::vector<int32_t> const& promptTokenIds, int32_t prefillStart,
    int32_t imageTokenId, int32_t audioTokenId, Sources const& sources, cudaStream_t stream)
{
    SlotMedia& media = slotMedia(slot);
    media = SlotMedia{};
    int32_t const promptLength = static_cast<int32_t>(promptTokenIds.size());
    ELLM_CHECK(prefillStart >= 0 && prefillStart <= promptLength,
        "CompactMediaReservation: the prefill start lies outside the prompt");
    size_t const rowBytes = static_cast<size_t>(mShape.hiddenSize) * sizeof(half);

    // One media kind: count its positions, check the sources cover them, record the region row of every position
    // from the prefill start on and copy those rows (they are contiguous in the sources, after the skipped ones).
    auto const stageKind = [&](int32_t tokenId, int32_t rowsPerSlot, Tensor const* source,
                               std::vector<Tensor const*> const& companions, std::vector<Tensor*> const& regions,
                               std::string const& kind, std::vector<int32_t>& rows) {
        if (tokenId < 0)
        {
            return;
        }
        int64_t total = 0;
        int64_t skipped = 0;
        for (int32_t position = 0; position < promptLength; ++position)
        {
            if (promptTokenIds[static_cast<size_t>(position)] == tokenId)
            {
                ++total;
                skipped += position < prefillStart ? 1 : 0;
            }
        }
        if (total == 0)
        {
            return;
        }
        ELLM_CHECK(source != nullptr,
            "CompactMediaReservation: the prompt has " + kind + " positions but no " + kind
                + " encoder output was supplied");
        int64_t const needed = total - skipped;
        ELLM_CHECK(needed <= rowsPerSlot,
            "CompactMediaReservation: " + std::to_string(needed) + " " + kind + " rows exceed the "
                + std::to_string(rowsPerSlot) + " rows reserved per slot");
        checkSource(*source, total, mShape.hiddenSize, kind + " encoder output");
        for (Tensor const* companion : companions)
        {
            checkSource(*companion, total, mShape.hiddenSize, kind + " deepstack feature");
        }

        rows.assign(static_cast<size_t>(promptLength), -1);
        int32_t next = 0;
        for (int32_t position = prefillStart; position < promptLength; ++position)
        {
            if (promptTokenIds[static_cast<size_t>(position)] == tokenId)
            {
                rows[static_cast<size_t>(position)] = next++;
            }
        }
        if (needed == 0)
        {
            return;
        }
        size_t const regionOffset = static_cast<size_t>(slot) * static_cast<size_t>(rowsPerSlot) * rowBytes;
        size_t const sourceOffset = static_cast<size_t>(skipped) * rowBytes;
        size_t const bytes = static_cast<size_t>(needed) * rowBytes;
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(regions[0]->rawPointer()) + regionOffset,
            static_cast<char const*>(source->rawPointer()) + sourceOffset, bytes, cudaMemcpyDeviceToDevice, stream));
        for (size_t index = 0; index < companions.size(); ++index)
        {
            CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(regions[index + 1]->rawPointer()) + regionOffset,
                static_cast<char const*>(companions[index]->rawPointer()) + sourceOffset, bytes,
                cudaMemcpyDeviceToDevice, stream));
        }
    };

    std::vector<Tensor const*> deepstackSources;
    std::vector<Tensor*> imageRegions{&mImage};
    if (sources.image != nullptr)
    {
        ELLM_CHECK(static_cast<int32_t>(sources.deepstack.size()) == mShape.deepstackLayers,
            "CompactMediaReservation: expected " + std::to_string(mShape.deepstackLayers) + " deepstack features, got "
                + std::to_string(sources.deepstack.size()));
        for (size_t layer = 0; layer < sources.deepstack.size(); ++layer)
        {
            deepstackSources.push_back(&sources.deepstack[layer].get());
            imageRegions.push_back(&mDeepstack[layer]);
        }
    }
    stageKind(
        imageTokenId, mShape.imageRowsPerSlot, sources.image, deepstackSources, imageRegions, "image", media.imageRows);
    stageKind(audioTokenId, mShape.audioRowsPerSlot, sources.audio, {}, {&mAudio}, "audio", media.audioRows);
    media.staged = true;
}

void CompactMediaReservation::release(int32_t slot)
{
    slotMedia(slot) = SlotMedia{};
}

bool CompactMediaReservation::holds(int32_t slot) const
{
    return slotMedia(slot).staged;
}

bool CompactMediaReservation::mapSlice(int32_t slot, int32_t begin, int32_t length, int32_t* out)
{
    SlotMedia const& media = slotMedia(slot);
    ELLM_CHECK(length >= 0 && (length == 0 || out != nullptr), "CompactMediaReservation: invalid slice");
    std::fill(out, out + length, -1);
    if (!media.staged || length == 0)
    {
        return false;
    }
    if (!media.imageRows.empty())
    {
        mapMediaIndicesToReservationRows(media.imageRows, begin, length, slot, mShape.imageRowsPerSlot,
            static_cast<int64_t>(mShape.slots) * mShape.imageRowsPerSlot, out);
    }
    if (!media.audioRows.empty())
    {
        mAudioScratch.resize(static_cast<size_t>(length));
        mapMediaIndicesToReservationRows(media.audioRows, begin, length, slot, mShape.audioRowsPerSlot,
            static_cast<int64_t>(mShape.slots) * mShape.audioRowsPerSlot, mAudioScratch.data());
        for (int32_t index = 0; index < length; ++index)
        {
            if (mAudioScratch[static_cast<size_t>(index)] >= 0)
            {
                ELLM_CHECK(out[index] < 0, "CompactMediaReservation: a position is both an image and an audio row");
                out[index] = mAudioScratch[static_cast<size_t>(index)];
            }
        }
    }
    return std::any_of(out, out + length, [](int32_t row) { return row >= 0; });
}

} // namespace rt
} // namespace trt_edgellm
