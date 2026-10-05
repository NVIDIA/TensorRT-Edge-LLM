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

// The audio loader is the only place a request's bytes become samples. Its
// header states three post-conditions -- always mono, float32 in [-1, 1], and
// at the sample rate the caller asked for -- and MelExtractor refuses PCM that
// does not meet the third. The other two it cannot check, so a mixdown or a
// scaling that went wrong reaches the encoder as quiet or clipped audio.
//
// The tests build WAV containers byte by byte, so what was encoded is known
// exactly and can be compared against what came back. The one exception is the
// FLAC case at the end, which reads a real compressed file: see the comment
// there for why a synthesised container cannot stand in for it.

#include "runtime/audioLoader.h"

#include "runtime/audioUtils.h"
#include "scratchDir.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kSampleRate{16000};

//! The decoded samples as a vector, the shape the expectations below compare against.
std::vector<float> samplesOf(audio::AudioPCM const& pcm)
{
    float const* const data = pcm.samples->dataPointer<float>();
    return {data, data + pcm.numSamples()};
}

void appendBytes(std::vector<uint8_t>& out, void const* data, size_t size)
{
    auto const* first = static_cast<uint8_t const*>(data);
    out.insert(out.end(), first, first + size);
}

void appendTag(std::vector<uint8_t>& out, char const* tag)
{
    appendBytes(out, tag, 4);
}

template <typename T>
void appendLittleEndian(std::vector<uint8_t>& out, T value)
{
    appendBytes(out, &value, sizeof(value));
}

//! A 16-bit PCM WAV holding `samples` laid out interleaved by channel.
//!
//! Written here rather than through a helper so the encoded amplitudes are the
//! test's own input: the decoded values are compared against them directly.
std::vector<uint8_t> makeWav(std::vector<int16_t> const& interleaved, int32_t sampleRate, int16_t channels)
{
    auto const dataBytes = static_cast<uint32_t>(interleaved.size() * sizeof(int16_t));
    constexpr int16_t kBitsPerSample{16};
    auto const blockAlign = static_cast<int16_t>(channels * kBitsPerSample / 8);

    std::vector<uint8_t> wav;
    appendTag(wav, "RIFF");
    appendLittleEndian<uint32_t>(wav, 36U + dataBytes);
    appendTag(wav, "WAVE");

    appendTag(wav, "fmt ");
    appendLittleEndian<uint32_t>(wav, 16U); // PCM header size
    appendLittleEndian<int16_t>(wav, 1);    // format: PCM
    appendLittleEndian<int16_t>(wav, channels);
    appendLittleEndian<uint32_t>(wav, static_cast<uint32_t>(sampleRate));
    appendLittleEndian<uint32_t>(wav, static_cast<uint32_t>(sampleRate) * static_cast<uint32_t>(blockAlign));
    appendLittleEndian<int16_t>(wav, blockAlign);
    appendLittleEndian<int16_t>(wav, kBitsPerSample);

    appendTag(wav, "data");
    appendLittleEndian<uint32_t>(wav, dataBytes);
    appendBytes(wav, interleaved.data(), dataBytes);
    return wav;
}

//! Full-scale 16-bit amplitude, so a decode that normalizes correctly returns
//! values right at the documented [-1, 1] bounds.
constexpr int16_t kFullScale{32767};

std::vector<int16_t> makeRamp(int32_t count, int16_t amplitude)
{
    std::vector<int16_t> samples(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i)
    {
        double const phase = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(count);
        samples[static_cast<size_t>(i)] = static_cast<int16_t>(std::lround(std::sin(phase) * amplitude));
    }
    return samples;
}

class AudioLoaderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mDir = makeScratchDir("audioLoaderTests");
    }

    void TearDown() override
    {
        std::filesystem::remove_all(mDir);
    }

    std::filesystem::path writeFile(std::string const& name, std::vector<uint8_t> const& bytes)
    {
        auto const path = mDir / name;
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<char const*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        return path;
    }

    std::filesystem::path mDir;
};

// The base case: a mono file comes back sample for sample, scaled into the
// documented range. The amplitudes are asserted rather than just the count,
// because a loader that returned the right number of zeros would otherwise pass.
TEST_F(AudioLoaderTest, DecodesMonoPcmIntoTheDocumentedFloatRange)
{
    constexpr int32_t kSamples{256};
    auto const encoded = makeRamp(kSamples, kFullScale);
    auto const wav = makeWav(encoded, kSampleRate, /*channels=*/1);

    audio::AudioPCM pcm;
    ASSERT_TRUE(audio::loadAudioBytes(wav.data(), wav.size(), kSampleRate, pcm));

    EXPECT_EQ(pcm.sampleRate, kSampleRate);
    ASSERT_EQ(pcm.numSamples(), kSamples);
    auto const samples = samplesOf(pcm);
    for (size_t i = 0; i < samples.size(); ++i)
    {
        EXPECT_NEAR(samples[i], static_cast<float>(encoded[i]) / 32768.0F, 1e-4) << "sample " << i;
    }
    EXPECT_TRUE(
        std::all_of(samples.begin(), samples.end(), [](float value) { return value >= -1.0F && value <= 1.0F; }));
}

// The header promises mono output regardless of the input's channel count, and
// says multi-channel is mixdown-averaged. Averaging is asserted with channels
// that differ: a loader that took only the left channel, or summed without
// dividing, both produce plausible audio and neither matches the mean.
TEST_F(AudioLoaderTest, MixesMultipleChannelsDownToTheirAverage)
{
    constexpr int32_t kFrames{64};
    constexpr int16_t kLeft{20000};
    constexpr int16_t kRight{-4000};

    std::vector<int16_t> interleaved;
    for (int32_t frame = 0; frame < kFrames; ++frame)
    {
        interleaved.push_back(kLeft);
        interleaved.push_back(kRight);
    }
    auto const wav = makeWav(interleaved, kSampleRate, /*channels=*/2);

    audio::AudioPCM pcm;
    ASSERT_TRUE(audio::loadAudioBytes(wav.data(), wav.size(), kSampleRate, pcm));

    EXPECT_EQ(pcm.numChannels, 1);
    // One sample per frame, not per channel.
    ASSERT_EQ(pcm.numSamples(), kFrames);

    float const expected = (static_cast<float>(kLeft) + static_cast<float>(kRight)) / 2.0F / 32768.0F;
    auto const samples = samplesOf(pcm);
    for (size_t i = 0; i < samples.size(); ++i)
    {
        EXPECT_NEAR(samples[i], expected, 1e-3) << "frame " << i;
    }
}

// MelExtractor refuses PCM whose rate differs from its own, so resampling here
// is what makes a file recorded at any rate usable. The frame count has to
// follow the rate ratio; a loader that relabelled the rate without resampling
// would keep the original count and pitch-shift the audio.
TEST_F(AudioLoaderTest, ResamplesToTheRateTheCallerAskedFor)
{
    constexpr int32_t kSourceRate{8000};
    constexpr int32_t kSourceFrames{800};
    auto const wav = makeWav(makeRamp(kSourceFrames, kFullScale / 2), kSourceRate, /*channels=*/1);

    audio::AudioPCM pcm;
    ASSERT_TRUE(audio::loadAudioBytes(wav.data(), wav.size(), kSampleRate, pcm));

    EXPECT_EQ(pcm.sampleRate, kSampleRate);
    // 0.1 s of audio at twice the rate is twice as many samples. The tolerance
    // covers the resampler's filter delay, not the ratio itself.
    auto const expectedFrames = static_cast<double>(kSourceFrames) * kSampleRate / kSourceRate;
    EXPECT_NEAR(static_cast<double>(pcm.numSamples()), expectedFrames, expectedFrames * 0.05);
}

// Requesting the rate the file already carries must not resample it: that is
// the common path, and a needless pass through the resampler would both cost
// time and low-pass the signal.
TEST_F(AudioLoaderTest, LeavesTheFrameCountAloneWhenTheRateAlreadyMatches)
{
    constexpr int32_t kFrames{512};
    auto const wav = makeWav(makeRamp(kFrames, kFullScale / 2), kSampleRate, /*channels=*/1);

    audio::AudioPCM pcm;
    ASSERT_TRUE(audio::loadAudioBytes(wav.data(), wav.size(), kSampleRate, pcm));

    EXPECT_EQ(pcm.numSamples(), kFrames);
}

// Bytes that are not a container the decoder recognizes have to be reported as
// a failure. The header is explicit that `out` is unspecified on failure, so
// the return value is the only thing a caller may act on.
TEST_F(AudioLoaderTest, ReportsFailureForBytesThatAreNotAudio)
{
    std::vector<uint8_t> const notAudio(64, 0x7F);
    audio::AudioPCM pcm;
    EXPECT_FALSE(audio::loadAudioBytes(notAudio.data(), notAudio.size(), kSampleRate, pcm));

    EXPECT_FALSE(audio::loadAudioBytes(nullptr, 0, kSampleRate, pcm));

    // A WAV header truncated before its samples is the shape a partial upload
    // arrives in, and is likewise rejected rather than decoded as silence.
    auto truncated = makeWav(makeRamp(64, kFullScale), kSampleRate, /*channels=*/1);
    truncated.resize(20);
    EXPECT_FALSE(audio::loadAudioBytes(truncated.data(), truncated.size(), kSampleRate, pcm));
}

// The file path is the offline entry point and has to agree with the byte
// entry point the server uses, or the two ways of submitting the same audio
// produce different features.
TEST_F(AudioLoaderTest, LoadingFromAFileMatchesLoadingTheSameBytes)
{
    auto const wav = makeWav(makeRamp(256, kFullScale / 2), kSampleRate, /*channels=*/1);
    auto const path = writeFile("tone.wav", wav);

    audio::AudioPCM fromBytes;
    audio::AudioPCM fromFile;
    ASSERT_TRUE(audio::loadAudioBytes(wav.data(), wav.size(), kSampleRate, fromBytes));
    ASSERT_TRUE(audio::loadAudioFile(path, kSampleRate, fromFile));

    EXPECT_EQ(fromFile.sampleRate, fromBytes.sampleRate);
    EXPECT_EQ(samplesOf(fromFile), samplesOf(fromBytes));
}

TEST_F(AudioLoaderTest, ReportsFailureForAMissingFile)
{
    audio::AudioPCM pcm;
    EXPECT_FALSE(audio::loadAudioFile(mDir / "absent.wav", kSampleRate, pcm));
}

// The AudioData wrapper is what the runners receive. It has to carry the
// decoded PCM and record the rate it was decoded at, because nothing
// downstream re-reads the container to find out.
TEST_F(AudioLoaderTest, AudioDataCarriesThePcmAndTheRateItWasDecodedAt)
{
    auto const wav = makeWav(makeRamp(128, kFullScale / 2), /*sampleRate=*/8000, /*channels=*/1);
    auto const path = writeFile("tone.wav", wav);

    audioUtils::AudioData fromBytes;
    ASSERT_TRUE(audioUtils::loadAudioDataFromBytes(wav.data(), wav.size(), kSampleRate, fromBytes));
    ASSERT_NE(fromBytes.pcm, nullptr);
    EXPECT_EQ(fromBytes.sampleRate, kSampleRate);
    EXPECT_EQ(fromBytes.pcm->sampleRate, kSampleRate);
    EXPECT_GT(fromBytes.pcm->numSamples(), 0);

    audioUtils::AudioData fromFile;
    ASSERT_TRUE(audioUtils::loadAudioDataFromFile(path, kSampleRate, fromFile));
    ASSERT_NE(fromFile.pcm, nullptr);
    EXPECT_EQ(samplesOf(*fromFile.pcm), samplesOf(*fromBytes.pcm));
}

// A failed decode must leave the container without PCM attached rather than
// half-populated, so a caller that checks `pcm` instead of the return value
// still cannot proceed on nothing.
// A failed decode has to clear the container, not just report false.
//
// Asserted on an AudioData that already holds a successful decode, which is the
// only arrangement that can tell "cleared it" from "never wrote it": against a
// fresh container the pointer is null before the call, so the check passes for
// any implementation, including one that does nothing at all.
//
// The hazard is a caller that reuses one container across files and misses the
// return value. It would then hand the previous file's samples to the audio
// runner as this file's -- audible nonsense, with nothing in the logs.
TEST_F(AudioLoaderTest, DecodingFailureClearsAPreviouslyLoadedResult)
{
    auto const wav = makeWav(makeRamp(kSampleRate / 10, kFullScale), kSampleRate, /*channels=*/1);

    audioUtils::AudioData data;
    ASSERT_TRUE(audioUtils::loadAudioDataFromBytes(wav.data(), wav.size(), kSampleRate, data));
    ASSERT_NE(data.pcm, nullptr);
    ASSERT_GT(data.pcm->numSamples(), 0);

    std::vector<uint8_t> const notAudio(64, 0x00);
    EXPECT_FALSE(audioUtils::loadAudioDataFromBytes(notAudio.data(), notAudio.size(), kSampleRate, data));
    EXPECT_EQ(data.pcm, nullptr);
}

// The same guarantee on the file entry point, which decodes through a separate
// call and so has its own failure path to get wrong.
TEST_F(AudioLoaderTest, FileDecodingFailureAlsoClearsAPreviouslyLoadedResult)
{
    auto const wav = makeWav(makeRamp(kSampleRate / 10, kFullScale), kSampleRate, /*channels=*/1);
    auto const good = writeFile("good.wav", wav);
    auto const bad = writeFile("bad.wav", std::vector<uint8_t>(64, 0x00));

    audioUtils::AudioData data;
    ASSERT_TRUE(audioUtils::loadAudioDataFromFile(good, kSampleRate, data));
    ASSERT_NE(data.pcm, nullptr);

    EXPECT_FALSE(audioUtils::loadAudioDataFromFile(bad, kSampleRate, data));
    EXPECT_EQ(data.pcm, nullptr);
}

// Every test above decodes a WAV the test itself assembled, which cannot tell
// whether the other two formats the header advertises still decode at all.
// Format dispatch happens inside miniaudio, so nothing here would notice if
// audioLoader.cpp's `MA_NO_*` block grew an `MA_NO_FLAC` to trim object size --
// the loader would keep passing every WAV assertion and start rejecting every
// FLAC request in production.
//
// The fixture is the LibriSpeech clip the multimodal audio example ships:
// 16 kHz mono, 56080 frames. Asserting the exact count matters because FLAC
// arrives in independently coded blocks, so a decoder that stopped after the
// first one, or that trusted STREAMINFO without draining, would still return
// plausible audio.
//
// Given the FLAC clip the audio example ships
// When it is decoded at its own rate
// Then the whole stream comes back as mono float32 in the documented range
TEST_F(AudioLoaderTest, DecodesTheFlacContainerTheHeaderAdvertises)
{
    std::filesystem::path const flac
        = std::filesystem::path(PROJECT_ROOT_DIR) / "examples/multimodal/audio/6930-75918-0000.flac";
    ASSERT_TRUE(std::filesystem::exists(flac)) << flac;

    audio::AudioPCM pcm;
    ASSERT_TRUE(audio::loadAudioFile(flac, kSampleRate, pcm));

    constexpr size_t kFlacFrames{56080};
    EXPECT_EQ(static_cast<size_t>(pcm.numSamples()), kFlacFrames);
    EXPECT_EQ(pcm.sampleRate, kSampleRate);
    EXPECT_EQ(pcm.numChannels, 1);

    auto const samples = samplesOf(pcm);
    auto const [quietest, loudest] = std::minmax_element(samples.begin(), samples.end());
    EXPECT_GE(*quietest, -1.0F);
    EXPECT_LE(*loudest, 1.0F);
    // Speech, so it must not have decoded to silence -- the only other way to
    // satisfy the bounds above.
    EXPECT_GT(*loudest - *quietest, 0.1F);
}

// The requested-rate post-condition has to hold for a compressed source too:
// the resampler sits after the decoder, and a path that only wired it up for
// the raw-PCM case would pass every WAV test above.
//
// Given the same FLAC clip
// When a rate other than the file's own is requested
// Then the frame count follows the rate ratio
TEST_F(AudioLoaderTest, ResamplesFlacToTheRateTheCallerAskedFor)
{
    std::filesystem::path const flac
        = std::filesystem::path(PROJECT_ROOT_DIR) / "examples/multimodal/audio/6930-75918-0000.flac";
    ASSERT_TRUE(std::filesystem::exists(flac)) << flac;

    constexpr int32_t kHalfRate{kSampleRate / 2};
    audio::AudioPCM pcm;
    ASSERT_TRUE(audio::loadAudioFile(flac, kHalfRate, pcm));

    EXPECT_EQ(pcm.sampleRate, kHalfRate);
    constexpr double kExpectedFrames{56080.0 / 2.0};
    EXPECT_NEAR(static_cast<double>(pcm.numSamples()), kExpectedFrames, kExpectedFrames * 0.05);
}

} // namespace
