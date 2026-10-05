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

// MelExtractor reproduces a HuggingFace feature extractor numerically. Getting
// it wrong does not fail: audio still transcribes, just worse, which no
// pipeline test distinguishes from a weaker model.
//
// Every expectation below comes from the contract melSpectrogram.h states --
// the window definitions, the per-model factory descriptions, and the two
// post-normalize formulas are all written down there -- rather than from what
// the implementation currently returns. Where a formula would mean
// reimplementing the transform in the test, the assertion is a property the
// transform must have instead: monotonic mel axis, non-negative triangular
// filters, an exactly bounded dynamic range.

#include "runtime/melSpectrogram.h"

#include "common/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;
using namespace trt_edgellm::rt::audio;

namespace
{

constexpr int32_t kSampleRate{16000};

//! Copy `samples` into a host tensor that the PCM owns.
void attachSamples(AudioPCM& pcm, std::vector<float> const& samples)
{
    pcm.samples = std::make_shared<Tensor>(
        Coords{static_cast<int64_t>(samples.size())}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT);
    std::copy(samples.begin(), samples.end(), pcm.samples->dataPointer<float>());
}

//! One second of a pure tone at `frequencyHz`, which is what the mel axis
//! assertions below locate.
AudioPCM makeTone(float frequencyHz, int32_t samples, int32_t sampleRate = kSampleRate)
{
    AudioPCM pcm;
    pcm.sampleRate = sampleRate;
    pcm.numChannels = 1;
    std::vector<float> tone(static_cast<size_t>(samples));
    for (int32_t i = 0; i < samples; ++i)
    {
        tone[static_cast<size_t>(i)] = std::sin(
            2.0F * static_cast<float>(M_PI) * frequencyHz * static_cast<float>(i) / static_cast<float>(sampleRate));
    }
    attachSamples(pcm, tone);
    return pcm;
}

std::vector<float> hostValues(Tensor const& tensor)
{
    std::vector<float> values(static_cast<size_t>(tensor.getShape().volume()));
    std::memcpy(values.data(), tensor.rawPointer(), values.size() * sizeof(float));
    return values;
}

// ---------------------------------------------------------------------------
// Analysis window
// ---------------------------------------------------------------------------

// The endpoint is what names the two spellings apart: a symmetric window ends on
// a zero tap, a periodic one does not, because its last tap belongs to the next
// period. Whisper asks for periodic and Parakeet for symmetric, so confusing them
// silently changes the spectrum every model downstream was trained against.
//
// They differ everywhere, not only there -- the angle is divided by `len` and by
// `len - 1`, so 377 of a 400-tap window's taps differ and the widest gap, ~6e-3,
// is in the interior. The endpoint is asserted because it is the defining
// difference rather than the largest one.
//
// Given the two Hann spellings the supported models ask for
// When both windows are built
// Then only the symmetric one ends on zero
TEST(MelExtractorWindowTest, OnlyTheSymmetricHannReturnsToZeroAtItsLastTap)
{
    // The two differ everywhere, not only here: periodic divides the angle by `len` and symmetric by `len - 1`, so
    // for a 400-tap window 377 taps differ and the largest gap is ~6e-3 in the interior. The endpoint is asserted
    // instead because it is the *defining* difference rather than the largest one -- a symmetric window is the one
    // that closes back on zero -- and because it is the difference a mixed-up denominator cannot fake.

    MelExtractorConfig periodicConfig;
    periodicConfig.name = "periodic";
    periodicConfig.windowType = WindowType::kHannPeriodic;
    MelExtractor const periodic{periodicConfig};

    MelExtractorConfig symmetricConfig = periodicConfig;
    symmetricConfig.name = "symmetric";
    symmetricConfig.windowType = WindowType::kHannSymmetric;
    MelExtractor const symmetric{symmetricConfig};

    auto const& periodicTaps = periodic.window();
    auto const& symmetricTaps = symmetric.window();
    ASSERT_EQ(periodicTaps.size(), static_cast<size_t>(periodicConfig.winLength));
    ASSERT_EQ(symmetricTaps.size(), periodicTaps.size());

    // The periodic window's final tap is the one step it has not yet taken back
    // to zero, stated as the closed form rather than as "bigger than some
    // epsilon" -- it is only 6e-5 for a 400-tap window, so an arbitrary
    // threshold either passes for a zero or fails for the correct value.
    auto const length = static_cast<double>(periodicConfig.winLength);
    double const lastPeriodicTap = 0.5 - 0.5 * std::cos(2.0 * M_PI * (length - 1.0) / length);

    EXPECT_NEAR(periodicTaps.front(), 0.0F, 1e-6);
    EXPECT_NEAR(symmetricTaps.front(), 0.0F, 1e-6);
    EXPECT_NEAR(symmetricTaps.back(), 0.0F, 1e-6);
    // Tolerance set by float32 cancellation in `0.5 - 0.5 cos(x)` near x = 2*pi,
    // not by the assertion's needs: it is still under a percent of the expected
    // 6.2e-5, so a window that ended on zero fails decisively.
    EXPECT_NEAR(periodicTaps.back(), lastPeriodicTap, 1e-7);
    EXPECT_GT(lastPeriodicTap, 0.0);
}

// torch.hann_window(N, periodic=True) is 0.5 - 0.5 cos(2*pi*n/N). Asserting the
// closed form rather than a few sampled properties is what pins the window to
// the one HF used, since several bell shapes satisfy "starts at zero, peaks in
// the middle".
TEST(MelExtractorWindowTest, PeriodicHannMatchesItsClosedForm)
{
    MelExtractorConfig config;
    config.name = "hann";
    config.windowType = WindowType::kHannPeriodic;
    MelExtractor const extractor{config};

    auto const& taps = extractor.window();
    auto const length = static_cast<double>(config.winLength);
    for (size_t n = 0; n < taps.size(); ++n)
    {
        double const expected = 0.5 - 0.5 * std::cos(2.0 * M_PI * static_cast<double>(n) / length);
        EXPECT_NEAR(taps[n], expected, 1e-5) << "tap " << n;
    }
}

// Hamming does not reach zero at either end; that pedestal is what distinguishes
// it from Hann and is the only reason a caller picks it.
TEST(MelExtractorWindowTest, HammingKeepsANonZeroPedestalAtBothEnds)
{
    MelExtractorConfig config;
    config.name = "hamming";
    config.windowType = WindowType::kHammingSymmetric;
    MelExtractor const extractor{config};

    auto const& taps = extractor.window();
    ASSERT_FALSE(taps.empty());
    EXPECT_GT(taps.front(), 1e-3F);
    EXPECT_GT(taps.back(), 1e-3F);
}

// ---------------------------------------------------------------------------
// Mel filter bank
// ---------------------------------------------------------------------------

class MelFilterBankTest : public ::testing::Test
{
protected:
    //! A bank over a fixed 400-point FFT, so `bins()` is the width every assertion below indexes by. Only nMel
    //! varies: the tests that walk every band use fewer of them to keep the failure output readable.
    static MelExtractor makeBank(int32_t nMel)
    {
        MelExtractorConfig config;
        config.name = "bank";
        config.nFFT = kNFFT;
        config.nMel = nMel;
        return MelExtractor{config};
    }

    static constexpr int32_t kNFFT{400};

    static constexpr int32_t bins()
    {
        return kNFFT / 2 + 1;
    }

    //! One filter's weights across the spectrum bins.
    static std::vector<float> filterRow(std::vector<float> const& bank, int32_t bins, int32_t index)
    {
        auto const first = bank.begin() + static_cast<ptrdiff_t>(index) * bins;
        return std::vector<float>(first, first + bins);
    }

    static size_t peakBin(std::vector<float> const& row)
    {
        return static_cast<size_t>(std::distance(row.begin(), std::max_element(row.begin(), row.end())));
    }
};

// The bank multiplies a [bins] spectrum into [nMel] bands, so its shape is
// fixed by nFFT and nMel. A bank of the wrong width would either read past the
// spectrum or silently ignore its top end.
TEST_F(MelFilterBankTest, HasOneRowPerMelBandAndOneColumnPerSpectrumBin)
{
    constexpr int32_t kBands{80};

    EXPECT_EQ(makeBank(kBands).melFilterBank().size(), static_cast<size_t>(kBands) * bins());
}

// Triangular filters: no negative weight, and each band has a single peak with
// no energy outside its own contiguous support. A bank that lost its shape --
// wrapped edges, an off-by-one in the mel-to-bin mapping -- shows up as a row
// with energy on both sides of a gap.
//
// Given a mel filter bank
// When each band's weights are walked across the spectrum bins
// Then every band is non-negative and has exactly one contiguous run of support
TEST_F(MelFilterBankTest, EveryBandIsANonNegativeSingleLobedTriangle)
{
    constexpr int32_t kBands{40};

    auto const extractor = makeBank(kBands);
    auto const& bank = extractor.melFilterBank();
    ASSERT_EQ(bank.size(), static_cast<size_t>(kBands) * bins());

    for (int32_t band = 0; band < kBands; ++band)
    {
        auto const row = filterRow(bank, bins(), band);
        SCOPED_TRACE("band " + std::to_string(band));

        EXPECT_TRUE(std::all_of(row.begin(), row.end(), [](float w) { return w >= 0.0F; }));
        EXPECT_GT(std::accumulate(row.begin(), row.end(), 0.0F), 0.0F) << "band is entirely empty";

        // Count sign changes in the run-length of non-zero weights: a triangle
        // has exactly one run.
        int32_t runs = 0;
        bool inside = false;
        for (float const weight : row)
        {
            bool const active = weight > 0.0F;
            if (active && !inside)
            {
                ++runs;
            }
            inside = active;
        }
        EXPECT_EQ(runs, 1) << "band support is not contiguous";
    }
}

// Bands are ordered low frequency to high. This is the axis every downstream
// assertion and every trained model reads along, and reversing it produces a
// perfectly well-formed bank that means the opposite thing.
//
// Given a mel filter bank
// When the peak bin of each band is located
// Then the peaks advance with the band index and reach the top of the spectrum
TEST_F(MelFilterBankTest, BandCentresIncreaseWithBandIndex)
{
    constexpr int32_t kBands{40};

    auto const extractor = makeBank(kBands);
    auto const& bank = extractor.melFilterBank();

    size_t previous = 0;
    for (int32_t band = 0; band < kBands; ++band)
    {
        size_t const centre = peakBin(filterRow(bank, bins(), band));
        EXPECT_GE(centre, previous) << "band " << band;
        previous = centre;
    }
    // The last band must sit near the top of the spectrum, or the bank covers
    // only part of the band it was asked for.
    EXPECT_GT(previous, static_cast<size_t>(bins()) / 2);
}

// ---------------------------------------------------------------------------
// Per-model factories
// ---------------------------------------------------------------------------

// The factory descriptions in the header are the contract each model's HF
// parity rests on. A default that drifts here is not visible anywhere else:
// the extractor still runs and still produces a plausible spectrogram.
TEST(MelExtractorFactoryTest, WhisperMatchesItsDocumentedConfiguration)
{
    auto const whisper = makeWhisperExtractor();
    auto const& config = whisper.config();

    EXPECT_EQ(config.sampleRate, 16000);
    EXPECT_EQ(config.nFFT, 400);
    EXPECT_EQ(config.hopLength, 160);
    EXPECT_EQ(config.nMel, 128);
    EXPECT_EQ(config.melScale, MelScale::kSlaney);
    EXPECT_EQ(config.melNorm, MelNorm::kSlaney);
    EXPECT_EQ(config.logType, LogType::kLog10);
    EXPECT_EQ(config.postNormalize, PostNormalize::kWhisperClamp);
    EXPECT_EQ(config.layout, MelLayout::kMelTime);
    EXPECT_EQ(config.framePadding, FramePadding::kCenterReflect);
}

TEST(MelExtractorFactoryTest, ParakeetMatchesItsDocumentedConfiguration)
{
    auto const parakeetExtractor = makeParakeetExtractor();
    auto const& config = parakeetExtractor.config();

    EXPECT_EQ(config.sampleRate, 16000);
    EXPECT_EQ(config.nFFT, 512);
    EXPECT_EQ(config.hopLength, 160);
    EXPECT_EQ(config.nMel, 128);
    EXPECT_EQ(config.windowType, WindowType::kHannSymmetric);
    EXPECT_FLOAT_EQ(config.preemphCoeff, 0.97F);
    EXPECT_EQ(config.logFloorMode, LogFloorMode::kAdd);
    EXPECT_FLOAT_EQ(config.logFloor, std::ldexp(1.0F, -24));
    EXPECT_EQ(config.postNormalize, PostNormalize::kPerFeatureMeanStd);
    EXPECT_EQ(config.layout, MelLayout::kTimeMel);
}

// The header defines this extractor by difference: "identical to Parakeet
// except the model consumes raw log-mel". Asserting it as a difference means a
// future change to Parakeet has to be mirrored deliberately rather than
// forgotten.
//
// Given the Parakeet and Nemotron-ASR extractors
// When their configurations are compared field by field
// Then they differ in post-normalization and in nothing else
TEST(MelExtractorFactoryTest, NemotronAsrIsParakeetWithoutTheNormalizationStep)
{
    auto const parakeetExtractor = makeParakeetExtractor();
    auto const nemotronExtractor = makeNemotronAsrExtractor();
    auto const& parakeet = parakeetExtractor.config();
    auto const& nemotron = nemotronExtractor.config();

    EXPECT_EQ(nemotron.postNormalize, PostNormalize::kNone);
    EXPECT_NE(parakeet.postNormalize, nemotron.postNormalize);

    EXPECT_EQ(nemotron.sampleRate, parakeet.sampleRate);
    EXPECT_EQ(nemotron.nFFT, parakeet.nFFT);
    EXPECT_EQ(nemotron.hopLength, parakeet.hopLength);
    EXPECT_EQ(nemotron.winLength, parakeet.winLength);
    EXPECT_EQ(nemotron.nMel, parakeet.nMel);
    EXPECT_EQ(nemotron.windowType, parakeet.windowType);
    EXPECT_FLOAT_EQ(nemotron.preemphCoeff, parakeet.preemphCoeff);
    EXPECT_EQ(nemotron.melScale, parakeet.melScale);
    EXPECT_EQ(nemotron.melNorm, parakeet.melNorm);
    EXPECT_EQ(nemotron.logType, parakeet.logType);
    EXPECT_EQ(nemotron.logFloorMode, parakeet.logFloorMode);
    EXPECT_FLOAT_EQ(nemotron.logFloor, parakeet.logFloor);
    EXPECT_EQ(nemotron.layout, parakeet.layout);
    EXPECT_EQ(nemotron.framePadding, parakeet.framePadding);
}

// The string dispatch is what a config file reaches the factories through. An
// unknown tag has to fail rather than fall back, or a typo silently extracts
// features for the wrong model.
TEST(MelExtractorFactoryTest, DispatchByNameResolvesToTheSameConfigurationAsTheDirectFactory)
{
    auto const whisperByName = makeExtractorByName("whisper");
    auto const whisperDirect = makeWhisperExtractor();
    EXPECT_EQ(whisperByName.config().nFFT, whisperDirect.config().nFFT);

    auto const parakeetByName = makeExtractorByName("parakeet");
    auto const parakeetDirect = makeParakeetExtractor();
    EXPECT_EQ(parakeetByName.config().nFFT, parakeetDirect.config().nFFT);
    EXPECT_EQ(parakeetByName.config().layout, parakeetDirect.config().layout);

    EXPECT_THROW(static_cast<void>(makeExtractorByName("whispr")), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(makeExtractorByName("")), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

class MelExtractionTest : public ::testing::Test
{
protected:
    //! Mel band carrying the most energy, averaged over time. The index is what
    //! the tests compare; its absolute value depends on the mel scale and is
    //! deliberately not asserted.
    static size_t loudestBand(Tensor const& mel, MelLayout layout)
    {
        auto const values = hostValues(mel);
        auto const rows = mel.getShape()[0];
        auto const columns = mel.getShape()[1];
        int64_t const bands = layout == MelLayout::kMelTime ? rows : columns;
        int64_t const frames = layout == MelLayout::kMelTime ? columns : rows;

        std::vector<double> energy(static_cast<size_t>(bands), 0.0);
        for (int64_t band = 0; band < bands; ++band)
        {
            for (int64_t frame = 0; frame < frames; ++frame)
            {
                int64_t const index = layout == MelLayout::kMelTime ? band * frames + frame : frame * bands + band;
                energy[static_cast<size_t>(band)] += values[static_cast<size_t>(index)];
            }
        }
        return static_cast<size_t>(std::distance(energy.begin(), std::max_element(energy.begin(), energy.end())));
    }
};

// The mel axis runs low to high, so a higher tone has to land in a higher band.
// This is the cheapest assertion that the transform is a spectrogram at all
// rather than a correctly shaped array: it fails for a flipped axis, a
// transposed layout, or a filter bank applied to the wrong spectrum half.
//
// Given two pure tones an octave and more apart
// When both are extracted
// Then the higher tone's energy peaks in a higher mel band
TEST_F(MelExtractionTest, AHigherToneLandsInAHigherMelBand)
{
    auto extractor = makeWhisperExtractor();
    constexpr int32_t kSamples{kSampleRate};

    Tensor low;
    Tensor high;
    ASSERT_TRUE(extractor.extract(makeTone(200.0F, kSamples), low));
    ASSERT_TRUE(extractor.extract(makeTone(4000.0F, kSamples), high));

    EXPECT_LT(loudestBand(low, MelLayout::kMelTime), loudestBand(high, MelLayout::kMelTime));
}

// Whisper post-normalization is `max(x, x.max() - 8)` followed by `(x + 4) / 4`.
// Both steps together mean the output spans exactly 8/4 = 2.0 whenever the
// clamp binds, whatever the audio was. Dropping either step, or changing a
// constant, moves the range off 2.0.
//! Span of a mel tensor, which is what Whisper's post-normalization bounds.
float melRange(Tensor const& mel)
{
    auto const values = hostValues(mel);
    EXPECT_FALSE(values.empty());
    auto const [minimum, maximum] = std::minmax_element(values.begin(), values.end());
    return *maximum - *minimum;
}

// Whisper post-normalization is `max(x, x.max() - 8)` followed by `(x + 4) / 4`.
// The clamp puts a floor 8 below the peak and the divide scales by a quarter, so
// 2.0 is the widest the output can be however loud or quiet the audio was.
//
// Given audio of any level, including silence
// When the Whisper extractor's post-normalization runs
// Then the output spans at most 2.0
TEST_F(MelExtractionTest, WhisperNormalizationNeverLetsTheOutputRangeExceedTwo)
{
    auto extractor = makeWhisperExtractor();

    AudioPCM silence;
    silence.sampleRate = kSampleRate;
    silence.numChannels = 1;
    attachSamples(silence, std::vector<float>(static_cast<size_t>(kSampleRate), 0.0F));

    for (auto const& [label, pcm] : {std::pair{"tone", makeTone(1000.0F, kSampleRate)}, std::pair{"silence", silence}})
    {
        SCOPED_TRACE(label);
        Tensor mel;
        ASSERT_TRUE(extractor.extract(pcm, mel));
        EXPECT_LE(melRange(mel), 2.0F + 1e-4F);
    }
}

// The bound above is reached exactly when the clamp binds, which needs the raw
// log-mel to span at least the 8 the floor sits below the peak. A real tone does;
// that is the case the constants were chosen for. Asserting it separately from
// the bound keeps "at most 2" from being satisfied by an implementation that
// dropped the clamp and simply produced a narrower range.
//
// Given a tone whose raw log-mel range reaches the clamp's 8-decade floor
// When the Whisper extractor's post-normalization runs
// Then the output spans exactly 2.0 = 8/4
TEST_F(MelExtractionTest, WhisperNormalizationSpansExactlyTwoWhenTheClampBinds)
{
    auto extractor = makeWhisperExtractor();

    Tensor mel;
    ASSERT_TRUE(extractor.extract(makeTone(1000.0F, kSampleRate), mel));

    EXPECT_NEAR(melRange(mel), 2.0F, 1e-4);
}

// Digital silence is the counterexample that makes the split above necessary: its
// log-mel is constant, so the clamp never binds and the output collapses to a
// single value. An implementation asserting 2.0 unconditionally would be wrong
// here, and this is what pins that.
//
// Given digital silence, whose log-mel has no dynamic range for the clamp to bite on
// When the Whisper extractor's post-normalization runs
// Then the output is flat
TEST_F(MelExtractionTest, WhisperNormalizationLeavesSilenceWithNoRangeAtAll)
{
    auto extractor = makeWhisperExtractor();

    AudioPCM silence;
    silence.sampleRate = kSampleRate;
    silence.numChannels = 1;
    attachSamples(silence, std::vector<float>(static_cast<size_t>(kSampleRate), 0.0F));

    Tensor mel;
    ASSERT_TRUE(extractor.extract(silence, mel));

    EXPECT_NEAR(melRange(mel), 0.0F, 1e-4);
}

// The two conventions describe the same data transposed. Asserting that the
// extents swap keeps a layout change from being a silent reinterpretation of
// the same buffer.
//
// Given two extractors configured with opposite layouts
// When the same audio is extracted through both
// Then the two output shapes are transposes of each other
TEST_F(MelExtractionTest, LayoutDecidesWhichAxisIsTimeAndWhichIsMel)
{
    auto whisper = makeWhisperExtractor();
    auto parakeet = makeParakeetExtractor();
    auto const pcm = makeTone(1000.0F, kSampleRate);

    Tensor melTime;
    Tensor timeMel;
    ASSERT_TRUE(whisper.extract(pcm, melTime));
    ASSERT_TRUE(parakeet.extract(pcm, timeMel));

    ASSERT_EQ(melTime.getShape().getNumDims(), 2);
    ASSERT_EQ(timeMel.getShape().getNumDims(), 2);
    EXPECT_EQ(melTime.getShape()[0], whisper.config().nMel);
    EXPECT_EQ(timeMel.getShape()[1], parakeet.config().nMel);
    // Same audio and same hop, so both hold the same number of frames.
    EXPECT_EQ(melTime.getShape()[1], timeMel.getShape()[0]);
}

// Frames advance by one hop, so the frame count is proportional to the audio
// length. The header's own anchor for this is "30 s of 16 kHz audio at hop 160
// gives T = 3000", i.e. one frame per hop.
//
// Given audio lengths that differ by whole hops
// When each is extracted
// Then the frame count follows the length in hops
TEST_F(MelExtractionTest, FrameCountFollowsTheAudioLengthInHops)
{
    auto extractor = makeWhisperExtractor();
    int32_t const hop = extractor.config().hopLength;

    Tensor shortMel;
    Tensor longMel;
    ASSERT_TRUE(extractor.extract(makeTone(1000.0F, 100 * hop), shortMel));
    ASSERT_TRUE(extractor.extract(makeTone(1000.0F, 200 * hop), longMel));

    EXPECT_EQ(shortMel.getShape()[1], 100);
    EXPECT_EQ(longMel.getShape()[1], 200);
}

// A static time length exists so an engine with a baked-in sequence constant
// always receives the shape it was traced for. It has to hold in both
// directions: short audio is padded, long audio is truncated.
//
// Given an extractor configured to pad to a fixed number of frames
// When audio shorter and longer than that is extracted
// Then both come back at exactly that frame count
TEST_F(MelExtractionTest, StaticTimePaddingPinsTheFrameCountInBothDirections)
{
    constexpr int32_t kStaticFrames{150};

    auto const whisper = makeWhisperExtractor();
    MelExtractorConfig config = whisper.config();
    config.timePadding = TimePadding::kStaticPad;
    config.staticTimeLength = kStaticFrames;
    MelExtractor extractor{config};

    Tensor padded;
    Tensor truncated;
    ASSERT_TRUE(extractor.extract(makeTone(1000.0F, 50 * config.hopLength), padded));
    ASSERT_TRUE(extractor.extract(makeTone(1000.0F, 400 * config.hopLength), truncated));

    EXPECT_EQ(padded.getShape()[1], kStaticFrames);
    EXPECT_EQ(truncated.getShape()[1], kStaticFrames);
}

// The extractor cannot resample; the caller decodes at the rate it was built
// for. Extracting anyway would shift every frequency by the rate ratio, which
// is a plausible-looking spectrogram of the wrong audio.
TEST_F(MelExtractionTest, RefusesPcmRecordedAtADifferentSampleRate)
{
    auto extractor = makeWhisperExtractor();
    auto pcm = makeTone(1000.0F, kSampleRate, /*sampleRate=*/8000);

    Tensor mel;
    EXPECT_FALSE(extractor.extract(pcm, mel));
}

// Empty input has no frames to extract. Reporting failure rather than an empty
// tensor keeps the caller from feeding a zero-length sequence to the encoder.
TEST_F(MelExtractionTest, RefusesEmptyAudio)
{
    auto extractor = makeWhisperExtractor();
    AudioPCM empty;
    empty.sampleRate = kSampleRate;

    Tensor mel;
    EXPECT_FALSE(extractor.extract(empty, mel));
}

// ---------------------------------------------------------------------------
// Provider parity
// ---------------------------------------------------------------------------

// The assertions above pin the configuration and the shape of the transform,
// but not its arithmetic: an extractor with every documented field correct and
// a wrong filter-bank row, window scaling, or log floor passes all of them.
// These compare the whole mel matrix against the HuggingFace feature extractor
// each factory names as its reference, for one fixed broadband waveform.
//
// The goldens are regenerated by unittests/resources/gen_mel_parity_golden.py.
// nemotron_asr has no transformers counterpart and stays pinned by difference
// against parakeet above.

//! The waveform the goldens were generated from, rebuilt rather than stored:
//! `gen_mel_parity_golden.py` drives the same LCG, and 960 samples of it would
//! be a third of the golden file for input that is not a reference value at all.
//! Only the top 24 bits are used and the scale is a power of two, so every step
//! is exact in float and neither side depends on libm.
constexpr int32_t kWaveformSamples{960};

std::vector<float> makeWaveform(int32_t count)
{
    std::vector<float> samples(static_cast<size_t>(count));
    uint32_t state = 12345U;
    for (int32_t i = 0; i < count; ++i)
    {
        state = state * 1103515245U + 12345U;
        samples[static_cast<size_t>(i)] = static_cast<float>(static_cast<double>(state >> 8) * 0x1p-23 - 1.0);
    }
    return samples;
}

//! One `name rows cols` header followed by `rows` lines of `cols` values.
struct GoldenMatrix
{
    int64_t rows{0};
    int64_t cols{0};
    std::vector<float> values;

    float at(int64_t row, int64_t col) const
    {
        return values[static_cast<size_t>(row * cols + col)];
    }
};

std::map<std::string, GoldenMatrix> loadGoldens()
{
    std::string const path = std::string(PROJECT_ROOT_DIR) + "/unittests/resources/mel_parity_golden.txt";
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open()) << "cannot open " << path;

    std::map<std::string, GoldenMatrix> goldens;
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream header(line);
        std::string name;
        GoldenMatrix matrix;
        header >> name >> matrix.rows >> matrix.cols;
        matrix.values.reserve(static_cast<size_t>(matrix.rows * matrix.cols));
        for (int64_t row = 0; row < matrix.rows; ++row)
        {
            std::getline(file, line);
            std::istringstream values(line);
            float value{};
            while (values >> value)
            {
                matrix.values.push_back(value);
            }
        }
        goldens.emplace(name, std::move(matrix));
    }
    return goldens;
}

class MelParityTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mGoldens = loadGoldens();
        mPcm.sampleRate = kSampleRate;
        mPcm.numChannels = 1;
        attachSamples(mPcm, makeWaveform(kWaveformSamples));
    }

    //! Compare `mel` against the named golden element for element. The
    //! goldens are stored in the layout their factory emits, so a transposed
    //! output fails on the shape rather than silently comparing the wrong
    //! cells.
    //! `tolerance` sits a decade above the widest disagreement observed across
    //! all three presets, 1.5e-5, which is float32 round-off against a golden
    //! computed in double. It is still four decades below the smallest error
    //! the goldens are there to catch.
    static constexpr float kTolerance{1e-4F};

    void expectMatchesGolden(std::string const& name, Tensor const& mel, float tolerance)
    {
        auto const& golden = mGoldens.at(name);
        ASSERT_EQ(mel.getShape().getNumDims(), 2);
        ASSERT_EQ(mel.getShape()[0], golden.rows);
        ASSERT_EQ(mel.getShape()[1], golden.cols);

        auto const values = hostValues(mel);
        for (int64_t row = 0; row < golden.rows; ++row)
        {
            for (int64_t col = 0; col < golden.cols; ++col)
            {
                float const actual = values[static_cast<size_t>(row * golden.cols + col)];
                ASSERT_NEAR(actual, golden.at(row, col), tolerance) << name << " at [" << row << ", " << col << "]";
            }
        }
    }

    std::map<std::string, GoldenMatrix> mGoldens;
    AudioPCM mPcm;
};

// Given the waveform the goldens were generated from
// When the Whisper extractor runs
// Then every cell matches WhisperFeatureExtractor
TEST_F(MelParityTest, WhisperMatchesTheHuggingFaceFeatureExtractor)
{
    auto extractor = makeWhisperExtractor();
    Tensor mel;
    ASSERT_TRUE(extractor.extract(mPcm, mel));
    expectMatchesGolden("whisper", mel, kTolerance);
}

// Given the waveform the goldens were generated from
// When the Parakeet extractor runs
// Then every cell matches ParakeetFeatureExtractor
TEST_F(MelParityTest, ParakeetMatchesTheHuggingFaceFeatureExtractor)
{
    auto extractor = makeParakeetExtractor();
    Tensor mel;
    ASSERT_TRUE(extractor.extract(mPcm, mel));
    expectMatchesGolden("parakeet", mel, kTolerance);
}

// Given the waveform the goldens were generated from
// When the Gemma4 audio extractor runs
// Then every cell matches Gemma4AudioFeatureExtractor
TEST_F(MelParityTest, Gemma4MatchesTheHuggingFaceFeatureExtractor)
{
    auto extractor = makeGemma4AudioExtractor();
    Tensor mel;
    ASSERT_TRUE(extractor.extract(mPcm, mel));
    expectMatchesGolden("gemma4", mel, kTolerance);
}

} // namespace
