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

#include "runtime/gr00tRuntime.h"

#include "common/cudaUtils.h"
#include "common/logger.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace trt_edgellm
{
namespace gr00t
{

namespace
{

using Clock = std::chrono::steady_clock;

constexpr int32_t kNbStages = 3; //!< observation, vl_backbone, action

double millisecondsBetween(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

Gr00tRuntime::Gr00tRuntime(std::string const& engineDir, cudaStream_t stream)
    : mStream(stream)
    , mPolicy(engineDir)
    , mRunner(engineDir, stream)
    , mVisual(engineDir, mRunner.getConfig().maxVlLen, mRunner.getConfig().headDim, mPolicy.vision().shortestEdge,
          mPolicy.vision().cropFraction, stream)
{
    for (auto& event : mEvents)
    {
        CUDA_CHECK(cudaEventCreate(&event));
    }
}

Gr00tRuntime::~Gr00tRuntime() noexcept
{
    for (auto& event : mEvents)
    {
        if (event != nullptr)
        {
            cudaEventDestroy(event);
        }
    }
}

std::vector<Gr00tFrame> Gr00tRuntime::resolveViews(
    std::vector<Gr00tCameraView> const& cameras, std::vector<rt::imageUtils::ImageData>& decoded) const
{
    std::vector<std::string> const& keys = mPolicy.videoKeys();
    if (cameras.size() != keys.size())
    {
        throw std::invalid_argument("GR00T " + mPolicy.embodimentTag() + " takes " + std::to_string(keys.size())
            + " camera(s), got " + std::to_string(cameras.size()));
    }
    // A named view goes to its contract slot; unnamed views are taken in order.
    std::vector<Gr00tCameraView const*> ordered(keys.size(), nullptr);
    size_t unnamed = 0;
    for (auto const& view : cameras)
    {
        size_t slot = 0;
        if (view.name.empty())
        {
            while (unnamed < keys.size() && ordered[unnamed] != nullptr)
            {
                ++unnamed;
            }
            slot = unnamed;
        }
        else
        {
            auto const it = std::find(keys.begin(), keys.end(), view.name);
            if (it == keys.end())
            {
                throw std::invalid_argument("GR00T camera '" + view.name + "' is not one of the contract's cameras");
            }
            slot = static_cast<size_t>(it - keys.begin());
        }
        if (ordered[slot] != nullptr)
        {
            throw std::invalid_argument("GR00T camera '" + keys[slot] + "' is given twice");
        }
        ordered[slot] = &view;
    }

    std::vector<Gr00tFrame> frames;
    frames.reserve(keys.size());
    decoded.clear();
    decoded.reserve(keys.size());
    for (auto const* view : ordered)
    {
        if ((view->rgb != nullptr) == !view->imagePath.empty())
        {
            throw std::invalid_argument("GR00T camera '" + view->name + "' needs exactly one of imagePath and rgb");
        }
        if (view->rgb != nullptr)
        {
            frames.push_back({view->rgb, view->height, view->width});
            continue;
        }
        // Decoding from a path allocates pinned memory per call; adapters pass rgb.
        decoded.push_back(rt::imageUtils::loadRgbImageFromFile(view->imagePath));
        rt::imageUtils::ImageData const& image = decoded.back();
        frames.push_back({image.data(), image.height, image.width});
    }
    return frames;
}

Gr00tActionChunk Gr00tRuntime::infer(Gr00tObservation const& observation)
{
    auto const policyStart = Clock::now();
    CUDA_CHECK(cudaEventRecord(mEvents[0], mStream));
    Gr00tConfig const& config = mRunner.getConfig();

    if (static_cast<int32_t>(observation.state.size()) != mPolicy.state().totalDim())
    {
        throw std::invalid_argument("GR00T " + mPolicy.embodimentTag() + " state is "
            + std::to_string(mPolicy.state().totalDim()) + " wide, got " + std::to_string(observation.state.size()));
    }

    std::vector<rt::imageUtils::ImageData> decoded;
    std::vector<Gr00tFrame> const frames = resolveViews(observation.cameras, decoded);
    std::string const instruction = mPolicy.formalizeInstruction(observation.task);
    Gr00tVlInputs const& vl = mVisual.encode(frames, instruction, mStream);
    int32_t const vlLen = static_cast<int32_t>(vl.inputIds.size());
    CUDA_CHECK(cudaEventRecord(mEvents[1], mStream));

    mRunner.prefill(vl.inputsEmbeds, vl.deepstackEmbeds, vl.ropeCosSin, vl.positionIds, 1, vlLen, mStream);
    CUDA_CHECK(cudaEventRecord(mEvents[2], mStream));

    mRunner.setCrossMasks(vl.textMask, vl.imageMask, mStream);
    mRunner.setState(mPolicy.normalizeState(observation.state, config.stateDim), mStream);

    Gr00tActionChunk chunk;
    chunk.actions = mRunner.generate(mStream);
    CUDA_CHECK(cudaEventRecord(mEvents[3], mStream));
    // generate() already drained the stream; this only completes the timestamp.
    CUDA_CHECK(cudaEventSynchronize(mEvents[3]));

    chunk.modelHorizon = config.actionHorizon;
    chunk.modelActionDim = config.actionDim;
    chunk.horizon = mPolicy.horizon();
    chunk.robotActionDim = mPolicy.action().totalDim();
    chunk.robotActions = mPolicy.unnormalizeAction(chunk.actions, config.actionHorizon, config.actionDim);
    chunk.tokenIds = vl.inputIds;
    float stage[kNbStages]{};
    for (int32_t i = 0; i < kNbStages; ++i)
    {
        CUDA_CHECK(cudaEventElapsedTime(&stage[i], mEvents[i], mEvents[i + 1]));
    }
    chunk.timings.observationMs = stage[0];
    chunk.timings.backboneMs = stage[1];
    chunk.timings.actionMs = stage[2];
    chunk.timings.policyMs = millisecondsBetween(policyStart, Clock::now());
    return chunk;
}

} // namespace gr00t
} // namespace trt_edgellm
