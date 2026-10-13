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

//! Nvfp4A16BlackwellMoeRunner (Thor SM110 W4A16 routed MoE):
//! decode and grouped-GEMM prefill against an FP4-exact double-precision
//! reference, CUDA-graph replay, dispatch policy and shape gating.

#include "common/cudaUtils.h"
#include "kernels/moe/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeDispatchPolicy.h"
#include "kernels/moe/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeJitCompiler.h"
#include "kernels/moe/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeJitRunner.h"
#include "kernels/moe/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeRunner.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

namespace moe = nvfp4_a16_blackwell_moe;

#define ASSERT_CUDA(expr) ASSERT_EQ((expr), cudaSuccess) << cudaGetErrorString(cudaGetLastError())

constexpr int32_t kTileN{128};
constexpr int32_t kTileK{64};

float e2m1ToFloat(unsigned const code)
{
    static constexpr float kLevels[8]{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float const v = kLevels[code & 7U];
    return (code & 8U) ? -v : v;
}

float e4m3ToFloat(unsigned const code)
{
    int32_t const sign = (code & 0x80U) ? -1 : 1;
    int32_t const exponent = static_cast<int32_t>((code >> 3) & 0xFU);
    int32_t const mantissa = static_cast<int32_t>(code & 7U);
    float const value = exponent > 0 ? (1.0f + mantissa / 8.0f) * std::ldexp(1.0f, exponent - 7)
                                     : (mantissa / 8.0f) * std::ldexp(1.0f, -6);
    return sign * value;
}

//! One expert projection in BLACKWELL_MOE_N128_K64_V1 plus its dense reference (alpha excluded).
struct ExpertWeights
{
    std::vector<uint8_t> codes;  // [nPad/128][k/64][128][32]
    std::vector<uint8_t> scales; // [nPad/128][k/64][128][4]
    std::vector<float> dense;    // [nPad][k]
    float alpha{};
};

ExpertWeights makeExpert(int32_t const n, int32_t const nPad, int32_t const k, std::mt19937& rng)
{
    ExpertWeights e{};
    int32_t const kTiles = k / kTileK;
    e.codes.assign(static_cast<size_t>(nPad) * k / 2, 0);
    e.scales.assign(static_cast<size_t>(nPad) * k / 16, 0);
    e.dense.assign(static_cast<size_t>(nPad) * k, 0.0f);
    std::uniform_int_distribution<int32_t> codeDist(0, 255);
    std::uniform_int_distribution<int32_t> scaleDist(0x28, 0x3F);
    std::uniform_real_distribution<float> alphaDist(0.004f, 0.012f);
    e.alpha = alphaDist(rng);
    for (int32_t row = 0; row < n; ++row)
    {
        for (int32_t kb = 0; kb < kTiles; ++kb)
        {
            size_t const rowTile = (static_cast<size_t>(row / kTileN) * kTiles + kb) * kTileN + (row % kTileN);
            for (int32_t g = 0; g < 4; ++g)
            {
                uint8_t const scaleByte = static_cast<uint8_t>(scaleDist(rng));
                e.scales[rowTile * 4 + g] = scaleByte;
                for (int32_t j = 0; j < 8; ++j)
                {
                    uint8_t const codeByte = static_cast<uint8_t>(codeDist(rng));
                    // BLACKWELL_MOE_N128_K64_V1 row bytes carry the TMA 32B swizzle:
                    // rows with bit 2 set swap their 16-byte halves.
                    int32_t const logicalByte = g * 8 + j;
                    int32_t const storedByte = ((logicalByte / 16) ^ ((row >> 2) & 1)) * 16 + logicalByte % 16;
                    e.codes[rowTile * 32 + storedByte] = codeByte;
                    int32_t const kk = kb * kTileK + g * 16 + j * 2;
                    e.dense[static_cast<size_t>(row) * k + kk] = e2m1ToFloat(codeByte & 0xFU) * e4m3ToFloat(scaleByte);
                    e.dense[static_cast<size_t>(row) * k + kk + 1]
                        = e2m1ToFloat(codeByte >> 4) * e4m3ToFloat(scaleByte);
                }
            }
        }
    }
    return e;
}

struct Problem
{
    int32_t numExperts;
    int32_t topK;
    int32_t hidden;
    int32_t inter;
    int32_t interPad;
    float scaling{2.5f};
};

struct DeviceBuffer
{
    void* ptr{nullptr};
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes)
    {
        if (bytes > 0)
        {
            EXPECT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess);
        }
    }
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr(other.ptr)
    {
        other.ptr = nullptr;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    ~DeviceBuffer()
    {
        release();
    }
    void release() noexcept
    {
        if (ptr != nullptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    template <typename T>
    T* as() const
    {
        return static_cast<T*>(ptr);
    }
};

template <typename T>
void upload(DeviceBuffer& buffer, std::vector<T> const& host)
{
    ASSERT_CUDA(cudaMemcpy(buffer.ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
}

//! Synthetic MoE layer: weights for every expert, one token batch, and a CPU reference.
class MoeFixture
{
public:
    MoeFixture(Problem const& p, int32_t numTokens, uint32_t seed)
        : p_(p)
        , numTokens_(numTokens)
    {
        std::mt19937 rng(seed);
        std::vector<uint8_t> q1, s1, q2, s2;
        std::vector<float> g1, g2;
        for (int32_t e = 0; e < p.numExperts; ++e)
        {
            w1_.push_back(makeExpert(p.inter, p.interPad, p.hidden, rng));
            w2_.push_back(makeExpert(p.hidden, p.hidden, p.inter, rng));
            q1.insert(q1.end(), w1_.back().codes.begin(), w1_.back().codes.end());
            s1.insert(s1.end(), w1_.back().scales.begin(), w1_.back().scales.end());
            q2.insert(q2.end(), w2_.back().codes.begin(), w2_.back().codes.end());
            s2.insert(s2.end(), w2_.back().scales.begin(), w2_.back().scales.end());
            g1.push_back(w1_.back().alpha);
            g2.push_back(w2_.back().alpha);
        }
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uniform_real_distribution<float> biasDist(-0.05f, 0.05f);
        logits_.resize(static_cast<size_t>(numTokens) * p.numExperts);
        bias_.resize(p.numExperts);
        hidden_.resize(static_cast<size_t>(numTokens) * p.hidden);
        hiddenF_.resize(hidden_.size());
        for (auto& v : logits_)
        {
            v = normal(rng);
        }
        for (auto& v : bias_)
        {
            v = biasDist(rng);
        }
        for (size_t i = 0; i < hidden_.size(); ++i)
        {
            hidden_[i] = __float2half(normal(rng));
            hiddenF_[i] = __half2float(hidden_[i]);
        }
        dLogits_ = DeviceBuffer(logits_.size() * sizeof(float));
        dBias_ = DeviceBuffer(bias_.size() * sizeof(float));
        dHidden_ = DeviceBuffer(hidden_.size() * sizeof(half));
        dQ1_ = DeviceBuffer(q1.size());
        dS1_ = DeviceBuffer(s1.size());
        dG1_ = DeviceBuffer(g1.size() * sizeof(float));
        dQ2_ = DeviceBuffer(q2.size());
        dS2_ = DeviceBuffer(s2.size());
        dG2_ = DeviceBuffer(g2.size() * sizeof(float));
        dOut_ = DeviceBuffer(static_cast<size_t>(numTokens) * p.hidden * sizeof(half));
        upload(dLogits_, logits_);
        upload(dBias_, bias_);
        upload(dHidden_, hidden_);
        upload(dQ1_, q1);
        upload(dS1_, s1);
        upload(dG1_, g1);
        upload(dQ2_, q2);
        upload(dS2_, s2);
        upload(dG2_, g2);
        computeReference();
    }

    Nvfp4A16BlackwellMoeParams params(
        moe::Backend const backend, bool const enablePdl = false, int32_t const fc2PrefetchSlots = -1) const
    {
        Nvfp4A16BlackwellMoeParams params{};
        params.dtype = moe::DecodeDtype::kFP16;
        params.backend = backend;
        params.enablePdl = enablePdl;
        params.fc2PrefetchSlots = fc2PrefetchSlots;
        params.numTokens = numTokens_;
        params.numExperts = p_.numExperts;
        params.topK = p_.topK;
        params.hiddenSize = p_.hidden;
        params.interSize = p_.inter;
        params.interSizePadded = p_.interPad;
        params.nGroup = 1;
        params.topkGroup = 1;
        params.normTopkProb = true;
        params.routedScalingFactor = p_.scaling;
        params.routerLogits = dLogits_.as<float>();
        params.correctionBias = dBias_.as<float>();
        params.hiddenStates = dHidden_.ptr;
        params.fc1QWeights = dQ1_.ptr;
        params.fc1BlockScales = dS1_.ptr;
        params.fc1GlobalScales = dG1_.as<float>();
        params.fc2QWeights = dQ2_.ptr;
        params.fc2BlockScales = dS2_.ptr;
        params.fc2GlobalScales = dG2_.as<float>();
        params.output = dOut_.ptr;
        params.jit = &jitFor(params);
        return params;
    }

    //! The NVRTC bundle of this shape and knob combination, exactly as the plugin
    //! compiles it at engine build (one compile and one module per distinct key).
    Nvfp4A16BlackwellMoeJitRunner const& jitFor(Nvfp4A16BlackwellMoeParams const& params) const
    {
        Nvfp4A16BlackwellMoeJitKey const key = makeNvfp4A16BlackwellMoeJitKey(params);
        for (auto const& runner : jitRunners_)
        {
            if (runner->getKey() == key)
            {
                return *runner;
            }
        }
        auto runner = std::make_unique<Nvfp4A16BlackwellMoeJitRunner>();
        runner->load(compileNvfp4A16BlackwellMoeJitKernel(key));
        jitRunners_.push_back(std::move(runner));
        return *jitRunners_.back();
    }

    void zeroOutput()
    {
        ASSERT_CUDA(cudaMemset(dOut_.ptr, 0, static_cast<size_t>(numTokens_) * p_.hidden * sizeof(half)));
    }

    //! Compare device output against the reference: cosine and per-element tolerance.
    void expectMatchesReference(char const* label, float const cosMin = 0.999f) const
    {
        std::vector<half> out(static_cast<size_t>(numTokens_) * p_.hidden);
        ASSERT_CUDA(cudaMemcpy(out.data(), dOut_.ptr, out.size() * sizeof(half), cudaMemcpyDeviceToHost));
        double dot = 0.0, na = 0.0, nb = 0.0, maxErr = 0.0, maxRef = 0.0;
        size_t violations = 0;
        for (size_t i = 0; i < out.size(); ++i)
        {
            double const a = __half2float(out[i]);
            double const b = reference_[i];
            dot += a * b;
            na += a * a;
            nb += b * b;
            double const err = std::fabs(a - b);
            maxErr = std::max(maxErr, err);
            maxRef = std::max(maxRef, std::fabs(b));
            if (err > 0.02 * std::fabs(b) + 0.05)
            {
                ++violations;
            }
        }
        double const cosine = dot / (std::sqrt(na * nb) + 1e-30);
        EXPECT_GT(cosine, cosMin) << label;
        EXPECT_EQ(violations, 0U) << label << " max_abs_err=" << maxErr << " max_ref=" << maxRef;
    }

    std::vector<half> downloadOutput() const
    {
        std::vector<half> out(static_cast<size_t>(numTokens_) * p_.hidden);
        EXPECT_EQ(cudaMemcpy(out.data(), dOut_.ptr, out.size() * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
        return out;
    }

private:
    void computeReference()
    {
        reference_.assign(static_cast<size_t>(numTokens_) * p_.hidden, 0.0f);
        int32_t const E = p_.numExperts;
        for (int32_t t = 0; t < numTokens_; ++t)
        {
            std::vector<float> sigmoid(E), biased(E);
            for (int32_t e = 0; e < E; ++e)
            {
                sigmoid[e] = 1.0f / (1.0f + std::exp(-logits_[static_cast<size_t>(t) * E + e]));
                biased[e] = sigmoid[e] + bias_[e];
            }
            std::vector<int32_t> idx(p_.topK);
            std::vector<float> w(p_.topK);
            float sum = 0.0f;
            for (int32_t k = 0; k < p_.topK; ++k)
            {
                int32_t best = 0;
                for (int32_t e = 1; e < E; ++e)
                {
                    if (biased[e] > biased[best])
                    {
                        best = e;
                    }
                }
                idx[k] = best;
                w[k] = sigmoid[best];
                sum += sigmoid[best];
                biased[best] = -3.0e38f;
            }
            for (int32_t k = 0; k < p_.topK; ++k)
            {
                w[k] *= p_.scaling / sum;
                int32_t const e = idx[k];
                std::vector<float> a1(p_.inter, 0.0f);
                for (int32_t n = 0; n < p_.inter; ++n)
                {
                    double acc = 0.0;
                    float const* row = &w1_[e].dense[static_cast<size_t>(n) * p_.hidden];
                    for (int32_t kk = 0; kk < p_.hidden; ++kk)
                    {
                        acc += static_cast<double>(hiddenF_[static_cast<size_t>(t) * p_.hidden + kk]) * row[kk];
                    }
                    float v = std::max(0.0f, static_cast<float>(acc) * w1_[e].alpha);
                    a1[n] = __half2float(__float2half(v * v));
                }
                float const weight = w[k] * w2_[e].alpha;
                for (int32_t n = 0; n < p_.hidden; ++n)
                {
                    double acc = 0.0;
                    float const* row = &w2_[e].dense[static_cast<size_t>(n) * p_.inter];
                    for (int32_t kk = 0; kk < p_.inter; ++kk)
                    {
                        acc += static_cast<double>(a1[kk]) * row[kk];
                    }
                    reference_[static_cast<size_t>(t) * p_.hidden + n] += weight * static_cast<float>(acc);
                }
            }
        }
    }

    Problem p_;
    int32_t numTokens_;
    mutable std::vector<std::unique_ptr<Nvfp4A16BlackwellMoeJitRunner>> jitRunners_;
    std::vector<ExpertWeights> w1_, w2_;
    std::vector<float> logits_, bias_, hiddenF_, reference_;
    std::vector<half> hidden_;
    DeviceBuffer dLogits_, dBias_, dHidden_, dQ1_, dS1_, dG1_, dQ2_, dS2_, dG2_, dOut_;
};

//! Small shape: E=128 (plugin contract), tiny H/I so the CPU reference is fast; I=192 pads to 256.
constexpr Problem kSmall{128, 6, 256, 192, 256};
//! Nemotron 3.5 Lightning routed-MoE shape.
constexpr Problem kNemotron{128, 6, 2688, 1856, 1920};
//! Single routed slot: FC2's pre-wait weight staging clamps to one slot and the
//! FC2 loop has no global-streamed slot left.
constexpr Problem kSmallTopK1{128, 1, 256, 192, 256};
//! Wide intermediate: FC2 stages 8 K tiles per slot, so one pre-wait weight slot
//! (6 * 8 * 272 + 8 * 4608 B) already exceeds the 48 KB budget and the runner
//! falls back to 0 staged slots.
constexpr Problem kSmallWideInter{128, 6, 256, 4096, 4096};

class Nvfp4A16BlackwellMoeRunnerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int32_t const sm = getSMVersion();
        if (sm != 110)
        {
            GTEST_SKIP() << "Nvfp4A16BlackwellMoeRunner targets SM110 (Thor), got SM" << sm;
        }
        Nvfp4A16BlackwellMoeParams probe{};
        probe.numTokens = 1;
        probe.numExperts = 128;
        probe.topK = 6;
        probe.hiddenSize = 256;
        probe.interSize = 192;
        probe.interSizePadded = 256;
        if (!Nvfp4A16BlackwellMoeRunner::isSupported(sm, probe))
        {
            GTEST_SKIP() << "nvfp4_a16_blackwell_moe CuTe DSL group is not linked into this build";
        }
    }

    void runOnce(MoeFixture& fixture, moe::Backend const backend, cudaStream_t const stream,
        bool const enablePdl = false, int32_t const fc2PrefetchSlots = -1)
    {
        Nvfp4A16BlackwellMoeParams const params = fixture.params(backend, enablePdl, fc2PrefetchSlots);
        size_t const workspaceBytes = Nvfp4A16BlackwellMoeRunner::getWorkspaceSize(params);
        ASSERT_GT(workspaceBytes, 0U);
        DeviceBuffer workspace(workspaceBytes);
        ASSERT_CUDA(Nvfp4A16BlackwellMoeRunner::prepare(params, stream));
        fixture.zeroOutput();
        ASSERT_CUDA(Nvfp4A16BlackwellMoeRunner::run(params, workspace.ptr, workspaceBytes, stream));
        ASSERT_CUDA(cudaStreamSynchronize(stream));
    }
};

TEST_F(Nvfp4A16BlackwellMoeRunnerTest, DecodeAutoMatchesReferenceSmallShape)
{
    // T=1 is the only token count the auto policy routes to the decode kernels.
    MoeFixture fixture(kSmall, 1, 11);
    runOnce(fixture, moe::Backend::kAuto, nullptr);
    fixture.expectMatchesReference("decode auto T=1 small");
}

TEST_F(Nvfp4A16BlackwellMoeRunnerTest, PrefillAutoMatchesReferenceSmallShape)
{
    // T=40 > kDecodeMaxTokens selects the grouped tcgen05 path (tn32).
    MoeFixture fixture(kSmall, 40, 12);
    runOnce(fixture, moe::Backend::kAuto, nullptr);
    fixture.expectMatchesReference("prefill auto T=40 small");
}

TEST_F(Nvfp4A16BlackwellMoeRunnerTest, ForcedBackendsAgreeSmallShape)
{
    // Both backends read the same weight buffer and must agree with the reference.
    MoeFixture fixture(kSmall, 6, 13);
    runOnce(fixture, moe::Backend::kDecode, nullptr);
    fixture.expectMatchesReference("forced decode T=6 small");
    runOnce(fixture, moe::Backend::kPrefill, nullptr);
    fixture.expectMatchesReference("forced prefill T=6 small");
}

TEST_F(Nvfp4A16BlackwellMoeRunnerTest, DecodeMatchesReferenceNemotronShape)
{
    MoeFixture fixture(kNemotron, 1, 21);
    runOnce(fixture, moe::Backend::kAuto, nullptr);
    fixture.expectMatchesReference("decode auto T=1 nemotron");
}

TEST_F(Nvfp4A16BlackwellMoeRunnerTest, PrefillMatchesReferenceNemotronShape)
{
    MoeFixture fixture(kNemotron, 20, 22);
    runOnce(fixture, moe::Backend::kAuto, nullptr);
    fixture.expectMatchesReference("prefill auto T=20 nemotron");
}

TEST_F(Nvfp4A16BlackwellMoeRunnerTest, PdlMatchesReferenceBothBackends)
{
    // Every kernel is launched with programmatic stream serialization; the
    // griddepcontrol waits must still order routing -> layout/gather -> FC1 ->
    // FC2 (and the reduces) on both paths and shapes.
    {
        // Decode is deterministic, so PDL off (no pre-wait staging) and PDL on
        // with FC2's two-slot shared-memory weight staging must agree bit for bit.
        MoeFixture fixture(kSmall, 1, 41);
        runOnce(fixture, moe::Backend::kAuto, nullptr, /*enablePdl=*/false, /*fc2PrefetchSlots=*/0);
        std::vector<half> const withoutPdl = fixture.downloadOutput();
        runOnce(fixture, moe::Backend::kAuto, nullptr, /*enablePdl=*/true, /*fc2PrefetchSlots=*/2);
        fixture.expectMatchesReference("pdl decode T=1 small, 2 staged slots");
        std::vector<half> const withPdl = fixture.downloadOutput();
        ASSERT_EQ(withoutPdl.size(), withPdl.size());
        for (size_t i = 0; i < withPdl.size(); ++i)
        {
            ASSERT_EQ(__half_as_ushort(withoutPdl[i]), __half_as_ushort(withPdl[i])) << "element " << i;
        }
    }
    {
        MoeFixture fixture(kSmall, 40, 42);
        runOnce(fixture, moe::Backend::kAuto, nullptr, /*enablePdl=*/true);
        fixture.expectMatchesReference("pdl prefill T=40 small");
    }
    {
        MoeFixture fixture(kNemotron, 1, 43);
        runOnce(fixture, moe::Backend::kAuto, nullptr, /*enablePdl=*/false, /*fc2PrefetchSlots=*/0);
        std::vector<half> const withoutPdl = fixture.downloadOutput();
        runOnce(fixture, moe::Backend::kAuto, nullptr, /*enablePdl=*/true, /*fc2PrefetchSlots=*/2);
        fixture.expectMatchesReference("pdl decode T=1 nemotron, 2 staged slots");
        std::vector<half> const withPdl = fixture.downloadOutput();
        ASSERT_EQ(withoutPdl.size(), withPdl.size());
        for (size_t i = 0; i < withPdl.size(); ++i)
        {
            ASSERT_EQ(__half_as_ushort(withoutPdl[i]), __half_as_ushort(withPdl[i])) << "element " << i;
        }
    }
    {
        // The 48 KB clamp drops the staging entirely for this shape.
        MoeFixture fixture(kSmallWideInter, 1, 47);
        runOnce(fixture, moe::Backend::kAuto, nullptr, /*enablePdl=*/true, /*fc2PrefetchSlots=*/2);
        fixture.expectMatchesReference("pdl decode T=1 wide intermediate (staging clamped to 0)");
    }
    {
        MoeFixture fixture(kNemotron, 20, 44);
        runOnce(fixture, moe::Backend::kAuto, nullptr, /*enablePdl=*/true);
        fixture.expectMatchesReference("pdl prefill T=20 nemotron");
    }
}

TEST_F(Nvfp4A16BlackwellMoeRunnerTest, TopK1MatchesReferenceDecodeAndForcedDecode)
{
    {
        MoeFixture fixture(kSmallTopK1, 1, 45);
        runOnce(fixture, moe::Backend::kAuto, nullptr, /*enablePdl=*/true, /*fc2PrefetchSlots=*/2);
        fixture.expectMatchesReference("pdl decode T=1 topK=1 (staging clamps to 1 slot)");
    }
    {
        MoeFixture fixture(kSmallTopK1, 5, 46);
        runOnce(fixture, moe::Backend::kDecode, nullptr, /*enablePdl=*/true, /*fc2PrefetchSlots=*/1);
        fixture.expectMatchesReference("pdl forced decode T=5 topK=1");
    }
}

TEST_F(Nvfp4A16BlackwellMoeRunnerTest, CudaGraphReplayMatchesEager)
{
    for (bool const enablePdl : {false, true})
        for (int32_t const numTokens : {1, 24})
        {
            MoeFixture fixture(kSmall, numTokens, 30 + numTokens);
            cudaStream_t stream{};
            ASSERT_CUDA(cudaStreamCreate(&stream));
            // PDL launches are captured as programmatic graph edges.
            Nvfp4A16BlackwellMoeParams const params = fixture.params(moe::Backend::kAuto, enablePdl);
            size_t const workspaceBytes = Nvfp4A16BlackwellMoeRunner::getWorkspaceSize(params);
            DeviceBuffer workspace(workspaceBytes);
            // Uncaptured warmup (module load) as the plugin's onShapeChange does.
            ASSERT_CUDA(Nvfp4A16BlackwellMoeRunner::prepare(params, stream));
            fixture.zeroOutput();
            ASSERT_CUDA(Nvfp4A16BlackwellMoeRunner::run(params, workspace.ptr, workspaceBytes, stream));
            ASSERT_CUDA(cudaStreamSynchronize(stream));
            std::vector<half> const eager = fixture.downloadOutput();

            cudaGraph_t graph{};
            cudaGraphExec_t graphExec{};
            ASSERT_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            ASSERT_CUDA(Nvfp4A16BlackwellMoeRunner::run(params, workspace.ptr, workspaceBytes, stream));
            ASSERT_CUDA(cudaStreamEndCapture(stream, &graph));
            ASSERT_CUDA(cudaGraphInstantiate(&graphExec, graph, 0));
            for (int32_t replay = 0; replay < 3; ++replay)
            {
                fixture.zeroOutput();
                ASSERT_CUDA(cudaGraphLaunch(graphExec, stream));
                ASSERT_CUDA(cudaStreamSynchronize(stream));
                fixture.expectMatchesReference(enablePdl ? "graph replay (pdl)" : "graph replay");
                std::vector<half> const replayed = fixture.downloadOutput();
                double maxDiff = 0.0;
                for (size_t i = 0; i < eager.size(); ++i)
                {
                    maxDiff = std::max(
                        maxDiff, std::fabs(static_cast<double>(__half2float(eager[i])) - __half2float(replayed[i])));
                }
                // Decode is deterministic; prefill FC2 scatter-adds in a data-dependent order (fp16 ulps).
                EXPECT_LT(maxDiff, numTokens <= moe::kDecodeMaxTokens ? 1e-6 : 0.05)
                    << "T=" << numTokens << " pdl=" << enablePdl;
            }
            ASSERT_CUDA(cudaGraphExecDestroy(graphExec));
            ASSERT_CUDA(cudaGraphDestroy(graph));
            ASSERT_CUDA(cudaStreamDestroy(stream));
        }
}

TEST(Nvfp4A16BlackwellMoeDispatchPolicyTest, LocksSealedPolicy)
{
    EXPECT_EQ(moe::resolveBackend(moe::Backend::kAuto, 1), moe::Backend::kDecode);
    EXPECT_EQ(moe::resolveBackend(moe::Backend::kAuto, moe::kDecodeMaxTokens), moe::Backend::kDecode);
    EXPECT_EQ(moe::resolveBackend(moe::Backend::kAuto, moe::kDecodeMaxTokens + 1), moe::Backend::kPrefill);
    EXPECT_EQ(moe::resolveBackend(moe::Backend::kPrefill, 1), moe::Backend::kPrefill);
    EXPECT_EQ(moe::resolveBackend(moe::Backend::kDecode, 4096), moe::Backend::kDecode);
    EXPECT_EQ(moe::selectTokenTile(2), moe::TokenTile::kTn8);
    EXPECT_EQ(moe::selectTokenTile(16), moe::TokenTile::kTn8);
    EXPECT_EQ(moe::selectTokenTile(17), moe::TokenTile::kTn16);
    EXPECT_EQ(moe::selectTokenTile(32), moe::TokenTile::kTn16);
    EXPECT_EQ(moe::selectTokenTile(33), moe::TokenTile::kTn32);
    EXPECT_EQ(moe::selectTokenTile(64), moe::TokenTile::kTn32);
    EXPECT_EQ(moe::selectTokenTile(256), moe::TokenTile::kTn32);
    EXPECT_EQ(moe::selectTokenTile(257), moe::TokenTile::kTn64);
    EXPECT_EQ(moe::selectTokenTile(512), moe::TokenTile::kTn64);
    EXPECT_EQ(moe::selectTokenTile(2048), moe::TokenTile::kTn64);
    EXPECT_EQ(moe::selectTokenTile(2049), moe::TokenTile::kTn128);
    // Nemotron 3.5 Lightning, T=2048: 12288 routed rows + 128 experts * 127 pad rows, tile-rounded.
    EXPECT_EQ(moe::maxRowsPadded(2048, 6, 128, 128), 28544);
    EXPECT_EQ(moe::maxRowsPadded(1, 6, 128, 32), 3968 + 32);
}

TEST(Nvfp4A16BlackwellMoeRunnerShapeTest, RejectsUnsupportedShapesAndDtypes)
{
    Nvfp4A16BlackwellMoeParams p{};
    p.numTokens = moe::kDecodeMaxTokens; // largest token count still on the decode kernels
    p.numExperts = 128;
    p.topK = 6;
    p.hiddenSize = 2688;
    p.interSize = 1856;
    p.interSizePadded = 1920;
    int32_t const sm = 110;
    // Baseline validity does not depend on hardware, only on the linked artifact.
    bool const baseline = Nvfp4A16BlackwellMoeRunner::isSupported(sm, p);
    Nvfp4A16BlackwellMoeParams bf16 = p;
    bf16.dtype = moe::DecodeDtype::kBF16;
    EXPECT_FALSE(Nvfp4A16BlackwellMoeRunner::isSupported(sm, bf16));
    Nvfp4A16BlackwellMoeParams badHidden = p;
    badHidden.hiddenSize = 2700;
    EXPECT_FALSE(Nvfp4A16BlackwellMoeRunner::isSupported(sm, badHidden));
    Nvfp4A16BlackwellMoeParams badInter = p;
    badInter.interSize = 1800;
    EXPECT_FALSE(Nvfp4A16BlackwellMoeRunner::isSupported(sm, badInter));
    Nvfp4A16BlackwellMoeParams badPad = p;
    badPad.interSizePadded = 1856;
    EXPECT_FALSE(Nvfp4A16BlackwellMoeRunner::isSupported(sm, badPad));
    Nvfp4A16BlackwellMoeParams badTopK = p;
    badTopK.topK = 33;
    EXPECT_FALSE(Nvfp4A16BlackwellMoeRunner::isSupported(sm, badTopK));
    EXPECT_FALSE(Nvfp4A16BlackwellMoeRunner::isSupported(120, p));
    EXPECT_FALSE(Nvfp4A16BlackwellMoeRunner::isSupported(87, p));
    if (baseline)
    {
        EXPECT_GT(Nvfp4A16BlackwellMoeRunner::getWorkspaceSize(p), 0U);
        // routing, FC1, FC2, FC2 split-K reduce, plus the FC1 reduce when the FC1
        // split-K is > 1; the runner honours the benchmark-only override
        // EDGELLM_MOE_DECODE_FC1_SPLITK, so pin the sealed value only when it is unset.
        if (std::getenv("EDGELLM_MOE_DECODE_FC1_SPLITK") == nullptr)
        {
            EXPECT_EQ(Nvfp4A16BlackwellMoeRunner::numGpuOps(p), 4 + (moe::kDecodeFc1SplitK > 1 ? 1 : 0));
        }
        Nvfp4A16BlackwellMoeParams prefill = p;
        prefill.numTokens = moe::kDecodeMaxTokens + 1;
        EXPECT_EQ(Nvfp4A16BlackwellMoeRunner::numGpuOps(prefill), 5); // routing, layout, gather, FC1, FC2
        prefill.numTokens = 128;
        EXPECT_EQ(Nvfp4A16BlackwellMoeRunner::numGpuOps(prefill), 5);
        EXPECT_GT(
            Nvfp4A16BlackwellMoeRunner::getWorkspaceSize(prefill), Nvfp4A16BlackwellMoeRunner::getWorkspaceSize(p));
    }
    EXPECT_EQ(Nvfp4A16BlackwellMoeRunner::getWorkspaceSize(badInter), 0U);
}

} // namespace
} // namespace kernel
} // namespace trt_edgellm
