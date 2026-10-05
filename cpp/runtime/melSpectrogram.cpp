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

/* Hand-rolled FFT adapted from https://github.com/ggml-org/whisper.cpp
 * SPDX-License-Identifier: MIT
 * SPDX-FileCopyrightText: Copyright (c) The ggml authors
 */

#include "melSpectrogram.h"

#include "common/logger.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace audio
{

namespace
{

constexpr float kPi = 3.14159265358979323846f;

//! Hand-rolled FFT: O(N log N) Cooley-Tukey radix-2 for even N, with O(N^2)
//! direct-DFT fallback for odd N. The values of N we hit in practice are
//! 400 (Whisper) and 512 (Parakeet); 512 stays in radix-2 the whole way
//! and 400 splits to dft(25) at the bottom -- cheap enough for ASR-grade
//! workloads (encoder forward dominates total latency).
//!
//! Twiddle factors come from a per-extractor sin/cos table sized to
//! ``cfg.nFFT`` (built once at init). Critical invariant: every recursion
//! level ``N`` (and the odd dft base ``M``) must divide ``table.size`` so
//! that ``step = table.size / N`` is exact -- otherwise the table-lookup
//! index drifts from the true angle. ``size = nFFT`` satisfies this because
//! ``nFFT = M * 2^k`` and every level divides ``nFFT`` exactly.
struct SinCosTable
{
    std::vector<float> sinT;
    std::vector<float> cosT;
    int32_t size{0};

    void build(int32_t newSize)
    {
        size = newSize;
        sinT.assign(static_cast<size_t>(newSize), 0.0f);
        cosT.assign(static_cast<size_t>(newSize), 0.0f);
        for (int32_t i = 0; i < newSize; ++i)
        {
            float const a = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(newSize);
            sinT[i] = std::sin(a);
            cosT[i] = std::cos(a);
        }
    }
};

//! Direct DFT for ``out_k = sum_n in[n] * exp(-j 2pi k n / N)``, used for odd N
//! and as the recursion base. Writes interleaved (re, im) into ``out`` (size
//! 2*N). Sin/cos values come from a table whose size is a multiple of N.
void dft(float const* in, int32_t n, float* out, SinCosTable const& tbl) noexcept
{
    int32_t const step = tbl.size / n;
    for (int32_t k = 0; k < n; ++k)
    {
        float re = 0.0f;
        float im = 0.0f;
        for (int32_t i = 0; i < n; ++i)
        {
            int32_t const idx = ((k * i) * step) % tbl.size;
            re += in[i] * tbl.cosT[idx];
            im -= in[i] * tbl.sinT[idx];
        }
        out[k * 2 + 0] = re;
        out[k * 2 + 1] = im;
    }
}

//! Cooley-Tukey radix-2 FFT. Recursive, in/out are real input → complex out
//! (interleaved (re, im), size 2*N). Falls back to ``dft()`` when N is odd.
//! Not noexcept: allocates ``std::vector<float>`` scratch buffers per recursion.
void fft(float const* in, int32_t n, float* out, SinCosTable const& tbl)
{
    if (n == 1)
    {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    if ((n & 1) != 0)
    {
        dft(in, n, out, tbl);
        return;
    }

    int32_t const half = n / 2;
    std::vector<float> even(half), odd(half);
    for (int32_t i = 0; i < half; ++i)
    {
        even[i] = in[2 * i + 0];
        odd[i] = in[2 * i + 1];
    }

    std::vector<float> evenOut(2 * half);
    std::vector<float> oddOut(2 * half);
    fft(even.data(), half, evenOut.data(), tbl);
    fft(odd.data(), half, oddOut.data(), tbl);

    int32_t const step = tbl.size / n;
    for (int32_t k = 0; k < half; ++k)
    {
        int32_t const idx = (k * step) % tbl.size;
        float const c = tbl.cosT[idx];
        float const s = -tbl.sinT[idx];

        float const twRe = c * oddOut[k * 2 + 0] - s * oddOut[k * 2 + 1];
        float const twIm = c * oddOut[k * 2 + 1] + s * oddOut[k * 2 + 0];

        out[k * 2 + 0] = evenOut[k * 2 + 0] + twRe;
        out[k * 2 + 1] = evenOut[k * 2 + 1] + twIm;
        out[(k + half) * 2 + 0] = evenOut[k * 2 + 0] - twRe;
        out[(k + half) * 2 + 1] = evenOut[k * 2 + 1] - twIm;
    }
}

//! Build a Hann window of ``len`` samples.
//!
//! periodic=True:  ``0.5 * (1 - cos(2 pi n / N))``         (HF Whisper default)
//! periodic=False: ``0.5 * (1 - cos(2 pi n / (N-1)))``      (HF Parakeet)
std::vector<float> hannWindow(int32_t len, bool periodic)
{
    // Degenerate sizes: symmetric (periodic=false) divides by ``len - 1``, so
    // len <= 1 would trip a div-by-zero. No real FE hits this (all three use
    // ``winLength = 400``); guard for robustness.
    if (len <= 1)
    {
        return std::vector<float>(static_cast<size_t>(std::max(len, 0)), 1.0f);
    }
    std::vector<float> w(len);
    float const denom = periodic ? static_cast<float>(len) : static_cast<float>(len - 1);
    for (int32_t i = 0; i < len; ++i)
    {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * kPi * static_cast<float>(i) / denom));
    }
    return w;
}

//! Build a Hamming window of ``len`` samples.
std::vector<float> hammingWindow(int32_t len, bool periodic)
{
    // See ``hannWindow`` for the len <= 1 rationale.
    if (len <= 1)
    {
        return std::vector<float>(static_cast<size_t>(std::max(len, 0)), 1.0f);
    }
    std::vector<float> w(len);
    float const denom = periodic ? static_cast<float>(len) : static_cast<float>(len - 1);
    for (int32_t i = 0; i < len; ++i)
    {
        w[i] = 0.54f - 0.46f * std::cos(2.0f * kPi * static_cast<float>(i) / denom);
    }
    return w;
}

std::vector<float> buildWindow(int32_t len, WindowType type)
{
    switch (type)
    {
    case WindowType::kHannPeriodic: return hannWindow(len, /*periodic=*/true);
    case WindowType::kHannSymmetric: return hannWindow(len, /*periodic=*/false);
    case WindowType::kHammingPeriodic: return hammingWindow(len, /*periodic=*/true);
    case WindowType::kHammingSymmetric: return hammingWindow(len, /*periodic=*/false);
    }
    return hannWindow(len, /*periodic=*/true);
}

//! Hz -> mel under one of the three HF mel scales.
float hzToMel(float hz, MelScale scale) noexcept
{
    switch (scale)
    {
    case MelScale::kHtk: return 2595.0f * std::log10(1.0f + hz / 700.0f);
    case MelScale::kKaldi: return 1127.0f * std::log(1.0f + hz / 700.0f);
    case MelScale::kSlaney:
    {
        // Mirror transformers/audio_utils.hertz_to_mel slaney branch.
        float constexpr fMin = 0.0f;
        float constexpr fSp = 200.0f / 3.0f;
        float constexpr minLogHz = 1000.0f;
        float const minLogMel = (minLogHz - fMin) / fSp;
        float const logstep = std::log(6.4f) / 27.0f;
        if (hz >= minLogHz)
        {
            return minLogMel + std::log(hz / minLogHz) / logstep;
        }
        return (hz - fMin) / fSp;
    }
    }
    return 0.0f;
}

float melToHz(float mel, MelScale scale) noexcept
{
    switch (scale)
    {
    case MelScale::kHtk: return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    case MelScale::kKaldi: return 700.0f * (std::exp(mel / 1127.0f) - 1.0f);
    case MelScale::kSlaney:
    {
        float constexpr fMin = 0.0f;
        float constexpr fSp = 200.0f / 3.0f;
        float constexpr minLogHz = 1000.0f;
        float const minLogMel = (minLogHz - fMin) / fSp;
        float const logstep = std::log(6.4f) / 27.0f;
        if (mel >= minLogMel)
        {
            return minLogHz * std::exp(logstep * (mel - minLogMel));
        }
        return fMin + fSp * mel;
    }
    }
    return 0.0f;
}

//! Build mel filter bank ``[nMel x nBins]`` row-major. Mirrors
//! ``transformers.audio_utils.mel_filter_bank`` for the given (scale, norm,
//! triangulariseInMelSpace) combination.
std::vector<float> buildMelFilterBank(int32_t sampleRate, int32_t nFFT, int32_t nMel, float fMin, float fMax,
    MelScale scale, MelNorm norm, bool triangulariseInMelSpace)
{
    int32_t const nBins = nFFT / 2 + 1;
    std::vector<float> filter(static_cast<size_t>(nMel) * nBins, 0.0f);

    float const melMin = hzToMel(fMin, scale);
    float const melMax = hzToMel(fMax, scale);

    std::vector<float> melPoints(nMel + 2);
    std::vector<float> hzPoints(nMel + 2);
    for (int32_t i = 0; i < nMel + 2; ++i)
    {
        float const t = static_cast<float>(i) / static_cast<float>(nMel + 1);
        melPoints[i] = melMin + t * (melMax - melMin);
        hzPoints[i] = melToHz(melPoints[i], scale);
    }

    // x-axis per FFT bin: Hz (default) or mel (Kaldi-style FEs). HF
    // computes ``fft_freqs = linspace(0, sr/2, nBins)``.
    std::vector<float> xPerBin(nBins);
    std::vector<float> const& anchor = triangulariseInMelSpace ? melPoints : hzPoints;
    for (int32_t k = 0; k < nBins; ++k)
    {
        float const hz
            = static_cast<float>(k) * (static_cast<float>(sampleRate) / 2.0f) / static_cast<float>(nBins - 1);
        xPerBin[k] = triangulariseInMelSpace ? hzToMel(hz, scale) : hz;
    }

    for (int32_t m = 0; m < nMel; ++m)
    {
        float const left = anchor[m];
        float const centre = anchor[m + 1];
        float const right = anchor[m + 2];
        for (int32_t k = 0; k < nBins; ++k)
        {
            float const x = xPerBin[k];
            float const up = (x - left) / (centre - left);
            float const down = (right - x) / (right - centre);
            float v = std::min(up, down);
            if (v < 0.0f)
            {
                v = 0.0f;
            }
            filter[static_cast<size_t>(m) * nBins + k] = v;
        }
        if (norm == MelNorm::kSlaney)
        {
            float const scaleNorm = 2.0f / (hzPoints[m + 2] - hzPoints[m]);
            for (int32_t k = 0; k < nBins; ++k)
            {
                filter[static_cast<size_t>(m) * nBins + k] *= scaleNorm;
            }
        }
    }
    return filter;
}

} // namespace

struct MelExtractor::Impl
{
    std::vector<float> windowFn;         //!< Length winLength.
    std::vector<float> melFilterStorage; //!< Used only when config.melFilter is null.
    float const* melFilterPtr{nullptr};
    int32_t nBins{0};
    std::vector<float> preempBuf; //!< Scratch for full-waveform preemph.
    SinCosTable sinCos;           //!< Twiddle factors, size = cfg.nFFT (built at init).
};

MelExtractor::MelExtractor(MelExtractorConfig cfg)
    : mImpl(std::make_unique<Impl>())
    , mConfig(std::move(cfg))
{
    if (mConfig.nFFT <= 0 || mConfig.hopLength <= 0 || mConfig.nMel <= 0 || mConfig.winLength <= 0)
    {
        throw std::invalid_argument("MelExtractor: nFFT / hopLength / nMel / winLength must be positive");
    }
    if (mConfig.winLength > mConfig.nFFT)
    {
        throw std::invalid_argument("MelExtractor: winLength > nFFT not supported (no centred zero-pad)");
    }

    mImpl->windowFn = buildWindow(mConfig.winLength, mConfig.windowType);
    mImpl->nBins = mConfig.nFFT / 2 + 1;
    mImpl->sinCos.build(mConfig.nFFT);

    float const fMax = (mConfig.maxFrequencyHz > 0.0f) ? mConfig.maxFrequencyHz : 0.5f * mConfig.sampleRate;

    if (mConfig.melFilter != nullptr)
    {
        mImpl->melFilterPtr = mConfig.melFilter;
    }
    else
    {
        mImpl->melFilterStorage = buildMelFilterBank(mConfig.sampleRate, mConfig.nFFT, mConfig.nMel,
            mConfig.minFrequencyHz, fMax, mConfig.melScale, mConfig.melNorm, mConfig.triangulariseInMelSpace);
        mImpl->melFilterPtr = mImpl->melFilterStorage.data();
    }
}

MelExtractor::MelExtractor() noexcept = default;
MelExtractor::~MelExtractor() = default;
MelExtractor::MelExtractor(MelExtractor&&) noexcept = default;
MelExtractor& MelExtractor::operator=(MelExtractor&&) noexcept = default;

std::vector<float> const& MelExtractor::melFilterBank() const noexcept
{
    static std::vector<float> const kEmpty;
    return mImpl ? mImpl->melFilterStorage : kEmpty;
}

std::vector<float> const& MelExtractor::window() const noexcept
{
    static std::vector<float> const kEmpty;
    return mImpl ? mImpl->windowFn : kEmpty;
}

bool MelExtractor::extract(AudioPCM const& pcm, Tensor& out)
{
    if (pcm.sampleRate != mConfig.sampleRate)
    {
        LOG_ERROR("MelExtractor[%s]: pcm.sampleRate=%d != config.sampleRate=%d", mConfig.name.c_str(), pcm.sampleRate,
            mConfig.sampleRate);
        return false;
    }
    if (pcm.numSamples() == 0)
    {
        LOG_ERROR("MelExtractor[%s]: empty PCM", mConfig.name.c_str());
        return false;
    }

    int32_t const nFFT = mConfig.nFFT;
    int32_t const hop = mConfig.hopLength;
    int32_t const nMel = mConfig.nMel;
    int32_t const nBins = mImpl->nBins;
    int32_t const winLen = mConfig.winLength;
    // ``winPadHead`` is the offset of the window inside the nFFT-sized FFT
    // input buffer. Centred (Whisper / Parakeet, torch.stft) or 0
    // (left-aligned: unfold + rfft(n=nFFT)).
    int32_t const winPadHead = mConfig.windowCentredInFft ? (nFFT - winLen) / 2 : 0;

    // ---- Source view of the waveform (with optional pre-emphasis applied).
    // Whisper (no preemph) and Parakeet (preemph applied to full waveform)
    // share the "preempBuf = preemph(samples)" path; per-frame preemph with
    // a separate post-scale is handled inline in the framing loop below.
    Tensor const& samplesTensor = *pcm.samples;
    float const* const pcmSamples = samplesTensor.dataPointer<float>();
    float const* srcPtr = pcmSamples;
    int32_t const numSamples = static_cast<int32_t>(pcm.numSamples());

    bool const preemphPerFrame = mConfig.preemphCoeff != 0.0f && mConfig.preemphPostScale != 0.0f;
    if (mConfig.preemphCoeff != 0.0f && !preemphPerFrame)
    {
        // HF Parakeet preemph: y[t] = x[t] - c * x[t-1], with y[0] = x[0]
        // (the first sample is *unchanged*; see
        // models/parakeet/feature_extraction_parakeet.py: cat([x[:1], x[1:] - c*x[:-1]])).
        mImpl->preempBuf.assign(pcmSamples, pcmSamples + numSamples);
        float const c = mConfig.preemphCoeff;
        for (int32_t i = numSamples - 1; i >= 1; --i)
        {
            mImpl->preempBuf[i] = pcmSamples[i] - c * pcmSamples[i - 1];
        }
        srcPtr = mImpl->preempBuf.data();
    }

    // ---- Optional centre-reflect padding (HF spectrogram(center=True,
    // pad_mode="reflect")). Materialised once into a local scratch.
    std::vector<float> centerBuf;
    int32_t srcLen = numSamples;
    if (mConfig.framePadding == FramePadding::kSemicausalZero)
    {
        // HF Gemma4: prepend winLength/2 zeros so the first frame is centred
        // at t=0; no right padding.
        int32_t const pad = winLen / 2;
        centerBuf.assign(static_cast<size_t>(numSamples) + pad, 0.0f);
        std::copy_n(srcPtr, numSamples, centerBuf.begin() + pad);
        srcPtr = centerBuf.data();
        srcLen = numSamples + pad;
    }
    else if (mConfig.framePadding == FramePadding::kCenterReflect || mConfig.framePadding == FramePadding::kCenterZero)
    {
        int32_t const pad = nFFT / 2;
        centerBuf.assign(static_cast<size_t>(numSamples) + 2 * pad, 0.0f);
        std::copy_n(srcPtr, numSamples, centerBuf.begin() + pad);
        if (mConfig.framePadding == FramePadding::kCenterReflect)
        {
            // HF reflect-pad: matches numpy.pad(x, pad, mode="reflect").
            for (int32_t i = 0; i < pad; ++i)
            {
                int32_t const j = std::min(i + 1, numSamples - 1);
                centerBuf[pad - 1 - i] = srcPtr[j];
            }
            for (int32_t i = 0; i < pad; ++i)
            {
                int32_t const j = std::max(numSamples - 2 - i, 0);
                centerBuf[pad + numSamples + i] = srcPtr[j];
            }
        }
        // kCenterZero: outer regions already zero from assign().
        srcPtr = centerBuf.data();
        srcLen = numSamples + 2 * pad;
    }

    // Frame count formula depends on framing convention.
    //
    //  - Centred (Whisper / Parakeet): the FFT buffer covers ``nFFT`` source
    //    samples, so ``nFrames = (srcLen - nFFT) / hop + 1``.
    //  - Left-aligned (unfold + rfft(n=nFFT)): each frame covers only
    //    ``winLen`` source samples (the tail of the FFT buffer is zero), so
    //    ``nFrames = (srcLen - winLen) / hop + 1``.
    int32_t frameSpanSamples = mConfig.windowCentredInFft ? nFFT : winLen;
    if (mConfig.framePadding == FramePadding::kSemicausalZero)
    {
        // HF Gemma4 unfolds frames of winLength+1 samples (the extra sample
        // only feeds the preemphasis filter and is dropped before the window).
        frameSpanSamples = winLen + 1;
    }
    int32_t nFrames = (srcLen >= frameSpanSamples) ? ((srcLen - frameSpanSamples) / hop + 1) : 1;
    // HF Whisper / Parakeet apply ``stft[..., :-1]`` before mel / log / post-norm.
    // Drop the trailing frame so post-normalize statistics (max-clamp, per-feature
    // mean/std) match HF byte-for-byte instead of drifting by O(1e-1) on the edge.
    if (mConfig.dropLastStftFrame && nFrames > 1)
    {
        nFrames -= 1;
    }

    int32_t outFrames = nFrames;
    if (mConfig.timePadding == TimePadding::kStaticPad)
    {
        if (mConfig.staticTimeLength <= 0)
        {
            LOG_ERROR("MelExtractor[%s]: kStaticPad requires staticTimeLength > 0", mConfig.name.c_str());
            return false;
        }
        outFrames = mConfig.staticTimeLength;
    }
    int32_t const framesToCompute = std::min(nFrames, outFrames);

    Coords const shape
        = (mConfig.layout == MelLayout::kMelTime) ? Coords({nMel, outFrames}) : Coords({outFrames, nMel});
    out = Tensor(shape, DeviceType::kCPU, nvinfer1::DataType::kFLOAT);
    float* outData = out.dataPointer<float>();
    std::fill_n(outData, static_cast<size_t>(nMel) * outFrames, 0.0f);

    std::vector<float> framedSamples(nFFT, 0.0f);
    // Hand-rolled FFT writes interleaved (re, im); 2 * nFFT floats covers it.
    std::vector<float> fftOut(static_cast<size_t>(2 * nFFT));
    std::vector<float> power(nBins);

    float const preempC = mConfig.preemphCoeff;
    float const preempScale = mConfig.preemphPostScale;

    for (int32_t t = 0; t < framesToCompute; ++t)
    {
        int32_t const sampleStart = t * hop;

        std::fill(framedSamples.begin(), framedSamples.end(), 0.0f);
        // Window covers source samples
        // ``[sampleStart + winPadHead, sampleStart + winPadHead + winLen)``,
        // mapped into ``framedSamples[winPadHead .. winPadHead + winLen)``
        // (matches torch.stft / librosa.stft / unfold+rfft(n=nFFT) semantics
        // depending on ``windowCentredInFft``).
        //
        // Per-frame pre-emph (preemphPerFrame=true) is intra-frame:
        // ``y[i] = (x[i] - c*x[i-1]) * scale``, with x[-1] = x[0] (HF rolls
        // then sets pos 0 to pos 1; after the roll, pos 1 holds frame x[0]).
        // ``samplePrev`` tracks the previous sample in the *current frame*.
        float samplePrev = 0.0f;
        for (int32_t i = 0; i < winLen; ++i)
        {
            int32_t const srcIdx = sampleStart + winPadHead + i;
            float sample = (srcIdx >= 0 && srcIdx < srcLen) ? srcPtr[srcIdx] : 0.0f;
            if (i == 0)
            {
                samplePrev = sample; // matches HF "frames_prev[:, :, 0] = frames_prev[:, :, 1]"
            }
            if (preemphPerFrame)
            {
                float const curr = sample;
                sample = (curr - preempC * samplePrev) * preempScale;
                samplePrev = curr;
            }
            else
            {
                samplePrev = sample;
            }
            framedSamples[winPadHead + i] = sample * mImpl->windowFn[i];
        }

        // Full complex FFT (we discard the upper-half conjugate symmetric bins
        // afterwards by reading only the first nBins = nFFT/2+1 entries).
        fft(framedSamples.data(), nFFT, fftOut.data(), mImpl->sinCos);

        for (int32_t k = 0; k < nBins; ++k)
        {
            float const re = fftOut[k * 2 + 0];
            float const im = fftOut[k * 2 + 1];
            float const p2 = re * re + im * im;
            power[k] = mConfig.magnitudeSpectrum ? std::sqrt(p2) : p2;
        }

        for (int32_t m = 0; m < nMel; ++m)
        {
            float acc = 0.0f;
            float const* filterRow = mImpl->melFilterPtr + static_cast<size_t>(m) * nBins;
            for (int32_t k = 0; k < nBins; ++k)
            {
                acc += filterRow[k] * power[k];
            }
            float const value = (mConfig.logFloorMode == LogFloorMode::kMax) ? std::max(acc, mConfig.logFloor)
                                                                             : acc + mConfig.logFloor;
            float const logVal = (mConfig.logType == LogType::kLog10) ? std::log10(value) : std::log(value);

            size_t const idx = (mConfig.layout == MelLayout::kMelTime) ? static_cast<size_t>(m) * outFrames + t
                                                                       : static_cast<size_t>(t) * nMel + m;
            outData[idx] = logVal;
        }
    }

    // ---- Post-normalize.
    if (mConfig.postNormalize == PostNormalize::kWhisperClamp)
    {
        // Whisper: log_spec = max(log_spec, log_spec.max() - 8); (x + 4) / 4.
        size_t const total = static_cast<size_t>(nMel) * outFrames;
        float maxVal = outData[0];
        for (size_t i = 1; i < total; ++i)
        {
            maxVal = std::max(maxVal, outData[i]);
        }
        float const floor = maxVal - 8.0f;
        for (size_t i = 0; i < total; ++i)
        {
            outData[i] = (std::max(outData[i], floor) + 4.0f) / 4.0f;
        }
    }
    else if (mConfig.postNormalize == PostNormalize::kPerFeatureMeanStd)
    {
        // Parakeet: per-mel mean/std over valid T (HF uses attention mask;
        // here ``framesToCompute`` is the valid extent).
        int32_t const validT = framesToCompute;
        if (validT > 0)
        {
            float constexpr kEps = 1e-5f;
            for (int32_t m = 0; m < nMel; ++m)
            {
                float sum = 0.0f;
                for (int32_t t = 0; t < validT; ++t)
                {
                    size_t const idx = (mConfig.layout == MelLayout::kMelTime) ? static_cast<size_t>(m) * outFrames + t
                                                                               : static_cast<size_t>(t) * nMel + m;
                    sum += outData[idx];
                }
                float const mean = sum / static_cast<float>(validT);
                float ss = 0.0f;
                for (int32_t t = 0; t < validT; ++t)
                {
                    size_t const idx = (mConfig.layout == MelLayout::kMelTime) ? static_cast<size_t>(m) * outFrames + t
                                                                               : static_cast<size_t>(t) * nMel + m;
                    float const d = outData[idx] - mean;
                    ss += d * d;
                }
                // Bessel-corrected, matching HF ParakeetFeatureExtractor's
                // `variance = ... / (features_lengths - 1)` and the GPU path in
                // `melStatsLnPerFeature`. Dividing by validT instead scales
                // every feature by sqrt(validT / (validT - 1)).
                int32_t const dof = validT > 1 ? validT - 1 : 1;
                float const denom = std::sqrt(ss / static_cast<float>(dof)) + kEps;
                for (int32_t t = 0; t < outFrames; ++t)
                {
                    size_t const idx = (mConfig.layout == MelLayout::kMelTime) ? static_cast<size_t>(m) * outFrames + t
                                                                               : static_cast<size_t>(t) * nMel + m;
                    outData[idx] = (outData[idx] - mean) / denom;
                }
            }
        }
    }

    return true;
}

MelExtractor makeWhisperExtractor()
{
    MelExtractorConfig cfg;
    cfg.name = "whisper";
    cfg.sampleRate = 16000;
    cfg.nFFT = 400;
    cfg.hopLength = 160;
    cfg.winLength = 400;
    cfg.nMel = 128; // Whisper-large-v3 / Qwen3-Omni / Qwen3-ASR all use 128.
    cfg.minFrequencyHz = 0.0f;
    cfg.maxFrequencyHz = 8000.0f;
    cfg.windowType = WindowType::kHannPeriodic;
    cfg.melScale = MelScale::kSlaney; // HF WhisperFeatureExtractor default.
    cfg.melNorm = MelNorm::kSlaney;
    cfg.logType = LogType::kLog10;
    cfg.logFloorMode = LogFloorMode::kMax;
    cfg.logFloor = 1e-10f;
    cfg.framePadding = FramePadding::kCenterReflect; // HF spectrogram(center=True, pad_mode="reflect").
    cfg.layout = MelLayout::kMelTime;
    cfg.postNormalize = PostNormalize::kWhisperClamp;
    cfg.timePadding = TimePadding::kNone;
    cfg.dropLastStftFrame = true; // HF Whisper drops stft[..., :-1] before mel.
    return MelExtractor(std::move(cfg));
}

MelExtractor makeGemma4AudioExtractor()
{
    // Matches HF Gemma4FeatureExtractor (USM-style front end): 20 ms periodic
    // Hann window, 10 ms hop, rfft(n=512) over the left-aligned frame,
    // magnitude spectrum, HTK mel without filter norm, ln(mel + 1e-3), no
    // post-normalization, semicausal time padding.
    MelExtractorConfig cfg;
    cfg.name = "gemma4-usm";
    cfg.sampleRate = 16000;
    cfg.nFFT = 512;
    cfg.hopLength = 160;
    cfg.winLength = 320;
    cfg.nMel = 128;
    cfg.minFrequencyHz = 0.0f;
    cfg.maxFrequencyHz = 8000.0f;
    cfg.windowType = WindowType::kHannPeriodic;
    cfg.windowCentredInFft = false;
    cfg.magnitudeSpectrum = true;
    cfg.melScale = MelScale::kHtk;
    cfg.melNorm = MelNorm::kNone;
    cfg.logType = LogType::kLn;
    cfg.logFloorMode = LogFloorMode::kAdd;
    cfg.logFloor = 1e-3f;
    cfg.framePadding = FramePadding::kSemicausalZero;
    cfg.layout = MelLayout::kMelTime;
    cfg.postNormalize = PostNormalize::kNone;
    cfg.timePadding = TimePadding::kNone;
    cfg.dropLastStftFrame = false;
    return MelExtractor(std::move(cfg));
}

MelExtractor makeParakeetExtractor()
{
    MelExtractorConfig cfg;
    cfg.name = "parakeet";
    cfg.sampleRate = 16000;
    cfg.nFFT = 512;
    cfg.hopLength = 160;
    cfg.winLength = 400;
    cfg.nMel = 128; // NVIDIA Nemotron-Omni sound_config.num_mel_bins.
    cfg.minFrequencyHz = 0.0f;
    cfg.maxFrequencyHz = 8000.0f;
    cfg.windowType = WindowType::kHannSymmetric; // periodic=False per HF Parakeet.
    cfg.melScale = MelScale::kSlaney;            // librosa.filters.mel default (htk=False).
    cfg.melNorm = MelNorm::kSlaney;
    cfg.preemphCoeff = 0.97f;    // HF default.
    cfg.preemphPostScale = 0.0f; // No per-frame scaling for Parakeet.
    cfg.logType = LogType::kLn;
    cfg.logFloorMode = LogFloorMode::kAdd;
    cfg.logFloor = 5.96046447754e-08f;            // 2^-24
    cfg.framePadding = FramePadding::kCenterZero; // HF torch.stft(pad_mode="constant").
    cfg.layout = MelLayout::kTimeMel;
    cfg.postNormalize = PostNormalize::kPerFeatureMeanStd;
    cfg.timePadding = TimePadding::kNone;
    cfg.dropLastStftFrame = true; // HF Parakeet's features_lengths = audio_len // hop excludes the trailing frame.
    return MelExtractor(std::move(cfg));
}

MelExtractor makeNemotronAsrExtractor()
{
    // Same front-end as Parakeet except the model consumes raw log-mel:
    // HF NemotronAsrStreamingFeatureExtractor never normalizes.
    MelExtractorConfig cfg;
    cfg.name = "nemotron_asr";
    cfg.sampleRate = 16000;
    cfg.nFFT = 512;
    cfg.hopLength = 160;
    cfg.winLength = 400;
    cfg.nMel = 128;
    cfg.minFrequencyHz = 0.0f;
    cfg.maxFrequencyHz = 8000.0f;
    cfg.windowType = WindowType::kHannSymmetric;
    cfg.melScale = MelScale::kSlaney;
    cfg.melNorm = MelNorm::kSlaney;
    cfg.preemphCoeff = 0.97f;
    cfg.preemphPostScale = 0.0f;
    cfg.logType = LogType::kLn;
    cfg.logFloorMode = LogFloorMode::kAdd;
    cfg.logFloor = 5.96046447754e-08f; // 2^-24
    cfg.framePadding = FramePadding::kCenterZero;
    cfg.layout = MelLayout::kTimeMel;
    cfg.postNormalize = PostNormalize::kNone;
    cfg.timePadding = TimePadding::kNone;
    cfg.dropLastStftFrame = true; // valid frames = audio_len // hop
    return MelExtractor(std::move(cfg));
}

MelExtractor makeExtractorByName(std::string const& feType)
{
    if (feType == "whisper")
    {
        return makeWhisperExtractor();
    }
    if (feType == "parakeet")
    {
        return makeParakeetExtractor();
    }
    if (feType == "nemotron_asr")
    {
        return makeNemotronAsrExtractor();
    }
    throw std::invalid_argument("Unknown feature extractor type: " + feType);
}

} // namespace audio
} // namespace rt
} // namespace trt_edgellm
