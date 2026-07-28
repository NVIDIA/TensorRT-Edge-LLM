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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/cosmosDenoiseRunner.h"

#include "common/logger.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{
namespace
{

std::string componentDir(std::filesystem::path const& engineRoot, char const* name)
{
    return (engineRoot / name).string();
}

bool componentExists(std::filesystem::path const& engineRoot, char const* name)
{
    return std::filesystem::exists(engineRoot / name / "config.json");
}

} // namespace

CosmosDenoiseRunner::CosmosDenoiseRunner(
    std::filesystem::path const& engineRoot, CosmosEngineConfig const& config, cudaStream_t stream)
    : mConfig(config)
{
    if (!componentExists(engineRoot, "embed") || !componentExists(engineRoot, "mot_backbone")
        || !componentExists(engineRoot, "denoise_head"))
    {
        throw std::runtime_error(
            "CosmosDenoiseRunner: missing embed/, mot_backbone/, or denoise_head/ under " + engineRoot.string());
    }

    mEmbedRunner = std::make_unique<EmbedRunner>(componentDir(engineRoot, "embed"), stream);
    mMotBackboneRunner = std::make_unique<MotBackboneRunner>(componentDir(engineRoot, "mot_backbone"), stream);
    mVisionHeadRunner = std::make_unique<DenoiseHeadRunner>(componentDir(engineRoot, "denoise_head"), stream);

    if (componentExists(engineRoot, "denoise_head_sound"))
    {
        mSoundHeadRunner = std::make_unique<DenoiseHeadRunner>(componentDir(engineRoot, "denoise_head_sound"), stream);
        LOG_INFO("CosmosDenoiseRunner: loaded optional denoise_head_sound.");
    }

    LOG_INFO("CosmosDenoiseRunner loaded from %s (embed + mot_backbone + denoise_head%s).", engineRoot.string().c_str(),
        mSoundHeadRunner ? " + denoise_head_sound" : "");
}

bool CosmosDenoiseRunner::isReady() const noexcept
{
    return mEmbedRunner && mMotBackboneRunner && mVisionHeadRunner;
}

bool CosmosDenoiseRunner::hasSoundPath() const noexcept
{
    return static_cast<bool>(mSoundHeadRunner);
}

int64_t CosmosDenoiseRunner::getRequiredContextMemorySize() const
{
    if (!isReady())
    {
        return 0;
    }

    int64_t size = std::max({mEmbedRunner->getRequiredContextMemorySize(),
        mMotBackboneRunner->getRequiredContextMemorySize(), mVisionHeadRunner->getRequiredContextMemorySize()});
    if (mSoundHeadRunner)
    {
        size = std::max(size, mSoundHeadRunner->getRequiredContextMemorySize());
    }
    return size;
}

bool CosmosDenoiseRunner::setContextMemory(rt::Tensor& sharedContextMemory)
{
    if (!isReady())
    {
        return true;
    }

    bool ok = mEmbedRunner->setContextMemory(sharedContextMemory)
        && mMotBackboneRunner->setContextMemory(sharedContextMemory)
        && mVisionHeadRunner->setContextMemory(sharedContextMemory);
    if (mSoundHeadRunner)
    {
        ok = ok && mSoundHeadRunner->setContextMemory(sharedContextMemory);
    }
    return ok;
}

bool CosmosDenoiseRunner::runDenoiseStep(CosmosTextPhase0 const& phase0, CosmosDenoiseLatents latents,
    CosmosFlowScheduler& visionScheduler, CosmosFlowScheduler* soundScheduler, cudaStream_t stream)
{
    if (latents.vision == nullptr)
    {
        LOG_ERROR("CosmosDenoiseRunner: vision latents are required.");
        return false;
    }

    float const timestep = visionScheduler.getTimestepValue();

    if (!mEmbedRunner->copyLatentsFrom(*latents.vision, stream))
    {
        LOG_ERROR("CosmosDenoiseRunner: embed copyLatentsFrom failed.");
        return false;
    }
    if (!mEmbedRunner->setTimestep(timestep, stream))
    {
        LOG_ERROR("CosmosDenoiseRunner: embed setTimestep failed.");
        return false;
    }
    if (!mEmbedRunner->embed(stream))
    {
        LOG_ERROR("CosmosDenoiseRunner: embed TRT failed.");
        return false;
    }

    if (!mMotBackboneRunner->copyUndSeqFrom(phase0.undSeq, stream))
    {
        LOG_ERROR("CosmosDenoiseRunner: backbone copyUndSeqFrom failed.");
        return false;
    }
    if (!mMotBackboneRunner->copyGenSeqFrom(mEmbedRunner->getGenSeq(), stream))
    {
        LOG_ERROR("CosmosDenoiseRunner: backbone copyGenSeqFrom failed.");
        return false;
    }
    if (!mMotBackboneRunner->copyRotaryFrom(phase0, stream))
    {
        LOG_ERROR("CosmosDenoiseRunner: backbone copyRotaryFrom failed.");
        return false;
    }
    if (!mMotBackboneRunner->runBackbone(stream))
    {
        LOG_ERROR("CosmosDenoiseRunner: mot_backbone TRT failed.");
        return false;
    }

    auto const& lastHidden = mMotBackboneRunner->getLastHiddenState();

    if (!mVisionHeadRunner->copyLastHiddenFrom(lastHidden, stream))
    {
        LOG_ERROR("CosmosDenoiseRunner: vision head copyLastHiddenFrom failed.");
        return false;
    }
    if (!mVisionHeadRunner->runHead(stream))
    {
        LOG_ERROR("CosmosDenoiseRunner: vision head TRT failed.");
        return false;
    }
    if (!visionScheduler.stepLatents(*latents.vision, mVisionHeadRunner->getPredLatents(), stream))
    {
        LOG_ERROR("CosmosDenoiseRunner: vision scheduler step failed.");
        return false;
    }

    if (latents.sound != nullptr && mSoundHeadRunner && soundScheduler != nullptr)
    {
        if (!mSoundHeadRunner->copyLastHiddenFrom(lastHidden, stream))
        {
            LOG_ERROR("CosmosDenoiseRunner: sound head copyLastHiddenFrom failed.");
            return false;
        }
        if (!mSoundHeadRunner->runHead(stream))
        {
            LOG_ERROR("CosmosDenoiseRunner: sound head TRT failed.");
            return false;
        }
        if (!soundScheduler->stepLatents(*latents.sound, mSoundHeadRunner->getPredLatents(), stream))
        {
            LOG_ERROR("CosmosDenoiseRunner: sound scheduler step failed.");
            return false;
        }
    }

    return true;
}

bool CosmosDenoiseRunner::sampleLatents(
    CosmosTextPhase0 const& phase0, CosmosDenoiseLatents latents, int32_t numInferenceSteps, cudaStream_t stream)
{
    if (!isReady())
    {
        LOG_ERROR("CosmosDenoiseRunner: denoise engines are not loaded.");
        return false;
    }

    if (latents.vision == nullptr)
    {
        LOG_ERROR("CosmosDenoiseRunner: vision latents pointer is null.");
        return false;
    }

    if (latents.sound != nullptr && !mSoundHeadRunner)
    {
        LOG_ERROR("CosmosDenoiseRunner: sound latents provided but denoise_head_sound is not loaded.");
        return false;
    }

    if (numInferenceSteps <= 0)
    {
        LOG_ERROR("CosmosDenoiseRunner: numInferenceSteps must be positive.");
        return false;
    }

    CosmosFlowScheduler visionScheduler(mConfig);
    visionScheduler.setTimesteps(numInferenceSteps);

    std::unique_ptr<CosmosFlowScheduler> soundScheduler;
    if (latents.sound != nullptr && mSoundHeadRunner)
    {
        soundScheduler = std::make_unique<CosmosFlowScheduler>(mConfig);
        soundScheduler->setTimesteps(numInferenceSteps);
    }

    LOG_INFO("CosmosDenoiseRunner: starting denoise loop (%d steps, sound=%s).", numInferenceSteps,
        latents.sound != nullptr ? "yes" : "no");

    for (int32_t step = 0; step < numInferenceSteps; ++step)
    {
        if (!runDenoiseStep(phase0, latents, visionScheduler, soundScheduler.get(), stream))
        {
            LOG_ERROR("CosmosDenoiseRunner: denoise step %d failed.", step);
            return false;
        }
    }

    LOG_INFO("CosmosDenoiseRunner: denoise loop complete.");
    return true;
}

bool CosmosDenoiseRunner::sampleVisionLatents(
    CosmosTextPhase0 const& phase0, rt::Tensor& visionLatents, int32_t numInferenceSteps, cudaStream_t stream)
{
    CosmosDenoiseLatents latents{};
    latents.vision = &visionLatents;
    return sampleLatents(phase0, latents, numInferenceSteps, stream);
}

} // namespace rt
} // namespace trt_edgellm
