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

//! Covers the borrowing entry audioUtils::wrapPcm: what it accepts, and that the SDK only reads the
//! samples a caller lends.

#include "common/tensor.h"
#include "runtime/audioLoader.h"
#include "runtime/audioUtils.h"
#include "runtime/melSpectrogram.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kSampleRate = 16000;

// A deterministic sawtooth in [-1, 1], enough signal for a non-degenerate mel.
std::vector<float> makeSignal(int64_t numSamples)
{
    std::vector<float> samples(static_cast<size_t>(numSamples));
    for (int64_t i = 0; i < numSamples; ++i)
    {
        samples[static_cast<size_t>(i)] = static_cast<float>((i * 37) % 4001 - 2000) / 2000.0F;
    }
    return samples;
}

// Non-owning view over caller memory, the shape a caller hands to wrapPcm.
std::shared_ptr<rt::Tensor> borrowSamples(std::vector<float>& samples)
{
    return std::make_shared<rt::Tensor>(samples.data(), rt::Coords{static_cast<int64_t>(samples.size())},
        rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT);
}

} // anonymous namespace

// wrapPcm publishes the caller's memory and allocates none of its own.
TEST(AudioPcmTest, WrapPcmBorrowsCallerMemory)
{
    std::vector<float> samples = makeSignal(kSampleRate);
    rt::audioUtils::AudioData const audio = rt::audioUtils::wrapPcm(borrowSamples(samples), kSampleRate);

    ASSERT_NE(audio.pcm, nullptr);
    EXPECT_EQ(audio.pcm->samples->rawPointer(), samples.data());
    EXPECT_FALSE(audio.pcm->samples->getOwnMemory());
    EXPECT_EQ(audio.pcm->numSamples(), static_cast<int64_t>(samples.size()));
    EXPECT_EQ(audio.pcm->sampleRate, kSampleRate);
    EXPECT_EQ(audio.pcm->numChannels, 1);
}

// Extractors read borrowed samples and write nothing back into them. Parakeet is covered alongside
// whisper because its pre-emphasis is the one stage that rewrites the waveform.
TEST(AudioPcmTest, ExtractionLeavesBorrowedSamplesUntouched)
{
    std::vector<float> samples = makeSignal(kSampleRate);
    std::vector<float> const original = samples;
    rt::audioUtils::AudioData const audio = rt::audioUtils::wrapPcm(borrowSamples(samples), kSampleRate);

    std::array<rt::audio::MelExtractor, 2> extractors{
        rt::audio::makeWhisperExtractor(), rt::audio::makeParakeetExtractor()};
    for (auto& extractor : extractors)
    {
        rt::Tensor mel;
        ASSERT_TRUE(extractor.extract(*audio.pcm, mel));
        ASSERT_GT(mel.getShape().volume(), 0);
        EXPECT_EQ(0, std::memcmp(samples.data(), original.data(), samples.size() * sizeof(float)));
    }
}

// wrapPcm takes non-empty one-dimensional host Float samples that point at memory, and nothing else.
TEST(AudioPcmTest, WrapPcmRejectsInvalidArguments)
{
    std::vector<float> samples = makeSignal(16);
    auto view = [&samples](rt::Coords const& shape, rt::DeviceType device, nvinfer1::DataType type) {
        return std::make_shared<rt::Tensor>(samples.data(), shape, device, type);
    };
    auto owned = [](rt::Coords const& shape, rt::DeviceType device, nvinfer1::DataType type) {
        return std::make_shared<rt::Tensor>(shape, device, type);
    };
    constexpr auto kCpu = rt::DeviceType::kCPU;
    constexpr auto kFloat = nvinfer1::DataType::kFLOAT;
    constexpr int64_t kTooMany = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;

    EXPECT_THROW(rt::audioUtils::wrapPcm(nullptr, kSampleRate), std::runtime_error);
    EXPECT_THROW(rt::audioUtils::wrapPcm(borrowSamples(samples), 0), std::runtime_error);
    EXPECT_THROW(rt::audioUtils::wrapPcm(borrowSamples(samples), -kSampleRate), std::runtime_error);
    EXPECT_THROW(
        rt::audioUtils::wrapPcm(view({static_cast<int64_t>(0)}, kCpu, kFloat), kSampleRate), std::runtime_error);
    EXPECT_THROW(rt::audioUtils::wrapPcm(view({4, 4}, kCpu, kFloat), kSampleRate), std::runtime_error);
    EXPECT_THROW(rt::audioUtils::wrapPcm(view({kTooMany}, kCpu, kFloat), kSampleRate), std::runtime_error);
    EXPECT_THROW(rt::audioUtils::wrapPcm(owned({16}, rt::DeviceType::kGPU, kFloat), kSampleRate), std::runtime_error);
    EXPECT_THROW(
        rt::audioUtils::wrapPcm(owned({16}, kCpu, nvinfer1::DataType::kHALF), kSampleRate), std::runtime_error);
    EXPECT_THROW(
        rt::audioUtils::wrapPcm(std::make_shared<rt::Tensor>(nullptr, rt::Coords{16}, kCpu, kFloat), kSampleRate),
        std::runtime_error);
}
