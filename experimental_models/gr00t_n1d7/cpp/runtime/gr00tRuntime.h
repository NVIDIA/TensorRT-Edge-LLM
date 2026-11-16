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

#include "action/gr00tPolicy.h"
#include "action/gr00tPolicyRunner.h"
#include "multimodal/gr00tVisualRunner.h"
#include "runtime/imageUtils.h"

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace gr00t
{

//! \brief One camera of an observation.
struct Gr00tCameraView
{
    //! Contract camera name from ``policy.json``'s ``video_keys``. Leave empty to
    //! place the view by position; naming it is checked, since two swapped views
    //! produce plausible actions rather than an error.
    std::string name;
    //! Exactly one source per view. An adapter already holding the frame sets
    //! \p rgb and skips the decode.
    std::string imagePath;
    //! Tightly packed row-major [height, width, 3] 8-bit RGB, borrowed for the call.
    unsigned char const* rgb{nullptr};
    int32_t height{0};
    int32_t width{0};
};

//! \brief What one policy call sees, in the robot's own units.
struct Gr00tObservation
{
    //! One entry per contract camera, all required.
    std::vector<Gr00tCameraView> cameras;
    //! Robot units, the embodiment's own width, in ``policy.json`` order.
    std::vector<float> state;
    std::string task;
};

//! \brief Where one policy call spent its time, in milliseconds.
struct Gr00tTimings
{
    double observationMs{0.0}; //!< decode, eval transform, chat template, tokenizer, tower, embedding assembly
    double backboneMs{0.0};    //!< vl_backbone prefill
    double actionMs{0.0};      //!< the denoise loop
    double policyMs{0.0};      //!< infer() entry through unnormalization
};

//! \brief One policy call's output.
struct Gr00tActionChunk
{
    //! Row-major [modelHorizon, modelActionDim], normalized and zero-padded as the
    //! action engine emits it.
    std::vector<float> actions;
    //! Row-major [horizon, robotActionDim], robot units.
    std::vector<float> robotActions;
    //! The prompt the backbone saw, so a chunk can be reproduced without re-deriving it.
    std::vector<int32_t> tokenIds;
    int32_t modelHorizon{0};
    int32_t modelActionDim{0};
    int32_t horizon{0};
    int32_t robotActionDim{0};
    Gr00tTimings timings;
};

//! \brief The GR00T policy chain: an observation in, robot commands out. Owns the vision
//! front end, the three engines and the embodiment contract. Batch 1: the contract carries
//! a batch axis, but a policy serves one observation at a time.
class Gr00tRuntime
{
public:
    Gr00tRuntime(std::string const& engineDir, cudaStream_t stream);
    ~Gr00tRuntime() noexcept;
    Gr00tRuntime(Gr00tRuntime const&) = delete;
    Gr00tRuntime& operator=(Gr00tRuntime const&) = delete;

    //! \brief Run one observation through the whole chain.
    //! \throws std::invalid_argument If the observation does not fit the contract.
    //! \throws std::runtime_error If an engine fails.
    Gr00tActionChunk infer(Gr00tObservation const& observation);

    Gr00tPolicy const& policy() const noexcept
    {
        return mPolicy;
    }

    //! \brief The engines behind infer(), for the knobs a caller sets once per
    //! process: noise, seed, denoise-loop graph capture.
    Gr00tPolicyRunner& runner() noexcept
    {
        return mRunner;
    }

private:
    //! Order the views by contract camera; views given as a path are decoded into \p decoded.
    std::vector<Gr00tFrame> resolveViews(
        std::vector<Gr00tCameraView> const& cameras, std::vector<rt::imageUtils::ImageData>& decoded) const;

    cudaStream_t mStream;
    Gr00tPolicy mPolicy;
    Gr00tPolicyRunner mRunner;
    Gr00tVisualRunner mVisual;
    //! Stage boundaries on mStream; read only after generate()'s D2H sync.
    cudaEvent_t mEvents[4]{};
};

} // namespace gr00t
} // namespace trt_edgellm
