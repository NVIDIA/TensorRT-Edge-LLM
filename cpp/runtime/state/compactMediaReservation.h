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

#include "common/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Per-slot copies of the encoder outputs that compact multimodal steps embed from.
//!
//! A prompt's media rows are produced once at its admission but consumed chunk by chunk over later steps, while
//! later admissions overwrite the encoders' output buffers. Each resident slot therefore owns a region of E rows in
//! every buffer: the image buffer and one buffer per deepstack layer (E = imageRowsPerSlot) and the audio buffer
//! (E = audioRowsPerSlot). Each buffer is one [slots * E, hiddenSize] FP16 tensor, slot s owning rows
//! [s * E, (s + 1) * E). A region holds, in prompt order, the rows of the media positions at or after the
//! sequence's prefill start; positions inside a reused prefix are never embedded again.
class CompactMediaReservation
{
public:
    struct Shape
    {
        int32_t slots{0};
        int32_t imageRowsPerSlot{0}; //!< 0 without a vision encoder.
        int32_t audioRowsPerSlot{0}; //!< 0 without an audio encoder.
        int32_t hiddenSize{0};
        int32_t deepstackLayers{0};
    };

    //! One request's encoder outputs, each [rows, hiddenSize] FP16 with one row per media position of the whole
    //! prompt, in prompt order. Only read during stage().
    struct Sources
    {
        Tensor const* image{nullptr};
        std::vector<std::reference_wrapper<Tensor const>> deepstack;
        Tensor const* audio{nullptr};
    };

    explicit CompactMediaReservation(Shape const& shape);
    //! deepstackRows() refers to this object's own buffers.
    CompactMediaReservation(CompactMediaReservation const&) = delete;
    CompactMediaReservation& operator=(CompactMediaReservation const&) = delete;

    Shape const& shape() const noexcept
    {
        return mShape;
    }

    //! Copy into @p slot's regions the rows of the media positions at or after @p prefillStart, and record where
    //! each prompt position's row lives. Replaces whatever the slot held.
    //! @throws std::runtime_error when a source does not cover the prompt's media positions, has the wrong shape or
    //!         type, or the rows needed exceed the slot's region.
    void stage(int32_t slot, std::vector<int32_t> const& promptTokenIds, int32_t prefillStart, int32_t imageTokenId,
        int32_t audioTokenId, Sources const& sources, cudaStream_t stream);

    //! Forget @p slot's media; its regions are left as they are and are never read until the next stage().
    void release(int32_t slot);

    //! Whether the sequence on @p slot was staged with media.
    bool holds(int32_t slot) const;

    //! Write the reservation rows of @p slot's prompt positions [begin, begin + length) into @p out, -1 at text
    //! positions; the global rows stay below slots * E.
    //! @return Whether any of the positions is a media position.
    bool mapSlice(int32_t slot, int32_t begin, int32_t length, int32_t* out);

    //! Image rows of every slot, [slots * imageRowsPerSlot, hiddenSize]; empty without a vision encoder.
    Tensor const& imageRows() const noexcept
    {
        return mImage;
    }

    //! Audio rows of every slot, [slots * audioRowsPerSlot, hiddenSize]; empty without an audio encoder.
    Tensor const& audioRows() const noexcept
    {
        return mAudio;
    }

    //! Deepstack rows of every slot, one tensor per layer with the image buffer's shape.
    OptionalInputTensors const& deepstackRows() const noexcept
    {
        return mDeepstackRows;
    }

private:
    struct SlotMedia
    {
        bool staged{false};
        //! Region row of every prompt position, -1 at the positions of the other kinds and before the prefill
        //! start. Empty when the prompt has no position of that kind.
        std::vector<int32_t> imageRows;
        std::vector<int32_t> audioRows;
    };

    SlotMedia& slotMedia(int32_t slot);
    SlotMedia const& slotMedia(int32_t slot) const;

    Shape mShape;
    Tensor mImage;
    Tensor mAudio;
    std::vector<Tensor> mDeepstack;
    //! References to mDeepstack's tensors, which never move after construction.
    OptionalInputTensors mDeepstackRows;
    std::vector<SlotMedia> mSlots;
    std::vector<int32_t> mAudioScratch;
};

} // namespace rt
} // namespace trt_edgellm
