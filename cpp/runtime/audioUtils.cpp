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

#include "audioUtils.h"

#include "audioLoader.h"
#include "common/checkMacros.h"

#include <cstdint>
#include <limits>

namespace trt_edgellm
{
namespace rt
{
namespace audioUtils
{

bool loadAudioDataFromBytes(uint8_t const* bytes, size_t size, int32_t targetSampleRate, AudioData& out)
{
    auto pcm = std::make_shared<audio::AudioPCM>();
    if (!audio::loadAudioBytes(bytes, size, targetSampleRate, *pcm))
    {
        // Drop whatever `out` was carrying. A caller that reuses one AudioData across loads and misses the return
        // value would otherwise hand the previous file's samples to the audio runner as if they were this one's.
        out.pcm.reset();
        return false;
    }
    out.pcm = std::move(pcm);
    out.sampleRate = targetSampleRate;
    return true;
}

bool loadAudioDataFromFile(std::filesystem::path const& path, int32_t targetSampleRate, AudioData& out)
{
    auto pcm = std::make_shared<audio::AudioPCM>();
    if (!audio::loadAudioFile(path, targetSampleRate, *pcm))
    {
        // Drop whatever `out` was carrying. A caller that reuses one AudioData across loads and misses the return
        // value would otherwise hand the previous file's samples to the audio runner as if they were this one's.
        out.pcm.reset();
        return false;
    }
    out.pcm = std::move(pcm);
    out.sampleRate = targetSampleRate;
    return true;
}

AudioData wrapPcm(std::shared_ptr<rt::Tensor> samples, int32_t sampleRate)
{
    ELLM_CHECK(samples != nullptr, "wrapPcm: samples is null.");
    ELLM_CHECK(samples->getDeviceType() == rt::DeviceType::kCPU, "wrapPcm: samples shall be host memory.");
    ELLM_CHECK(samples->getDataType() == nvinfer1::DataType::kFLOAT, "wrapPcm: samples shall be Float.");
    ELLM_CHECK(samples->getShape().getNumDims() == 1, "wrapPcm: samples shall be one-dimensional.");
    ELLM_CHECK(samples->getShape().volume() > 0, "wrapPcm: samples shall not be empty.");
    ELLM_CHECK(samples->getShape().volume() <= std::numeric_limits<int32_t>::max(),
        "wrapPcm: samples shall hold at most INT32_MAX samples.");
    ELLM_CHECK(samples->rawPointer() != nullptr, "wrapPcm: samples shall point at memory.");
    ELLM_CHECK(sampleRate > 0, "wrapPcm: sampleRate shall be positive.");

    auto pcm = std::make_shared<audio::AudioPCM>();
    pcm->samples = std::move(samples);
    pcm->sampleRate = sampleRate;

    AudioData out;
    out.pcm = std::move(pcm);
    out.sampleRate = sampleRate;
    return out;
}

} // namespace audioUtils
} // namespace rt
} // namespace trt_edgellm
