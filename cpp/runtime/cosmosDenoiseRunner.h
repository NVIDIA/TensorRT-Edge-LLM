/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/denoiseHeadRunner.h"
#include "runtime/embedRunner.h"
#include "runtime/motBackboneRunner.h"
#include "runtime/wfmRuntimeUtils.h"

#include <cuda_runtime.h>
#include <filesystem>
#include <memory>

namespace trt_edgellm
{
namespace rt
{

//! Orchestrates Cosmos denoise TRT engines + scheduler (ActionRunner analogue for WFM).
//!
//! Edge today: vision-only path (embed -> backbone -> head -> UniPC step).
//! Omni: optional sound head + independent sound scheduler; shared backbone per step.
class CosmosDenoiseRunner
{
public:
    //! \p engineRoot contains ``embed/``, ``mot_backbone/``, ``denoise_head/`` subdirs.
    //! Optional ``denoise_head_sound/`` enables the sound denoise path.
    explicit CosmosDenoiseRunner(
        std::filesystem::path const& engineRoot, CosmosEngineConfig const& config, cudaStream_t stream);

    ~CosmosDenoiseRunner() noexcept = default;

    bool isReady() const noexcept;
    bool hasSoundPath() const noexcept;

    int64_t getRequiredContextMemorySize() const;
    bool setContextMemory(rt::Tensor& sharedContextMemory);

    //! Run the denoise scheduler loop; updates vision and/or sound latents in place.
    bool sampleLatents(CosmosTextPhase0 const& phase0, CosmosDenoiseLatents latents, int32_t numInferenceSteps,
        cudaStream_t stream);

    //! Vision-only convenience wrapper.
    bool sampleVisionLatents(CosmosTextPhase0 const& phase0, rt::Tensor& visionLatents, int32_t numInferenceSteps,
        cudaStream_t stream);

private:
    bool runDenoiseStep(CosmosTextPhase0 const& phase0, CosmosDenoiseLatents latents,
        CosmosFlowScheduler& visionScheduler, CosmosFlowScheduler* soundScheduler, cudaStream_t stream);

    CosmosEngineConfig mConfig{};
    std::unique_ptr<EmbedRunner> mEmbedRunner;
    std::unique_ptr<MotBackboneRunner> mMotBackboneRunner;
    std::unique_ptr<DenoiseHeadRunner> mVisionHeadRunner;
    std::unique_ptr<DenoiseHeadRunner> mSoundHeadRunner;
};

} // namespace rt
} // namespace trt_edgellm
