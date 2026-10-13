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

#include "kernels/qsaIndexer/qsaIndexerKernels.h"
#include "common/checkMacros.h"
#include "common/tensor.h"
#include "kernels/qsaIndexer/qsaIndexerRunner.h"
#include "qsaIndexerReference.h"
#include "testUtils.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kWidth = kernel::kQSA_INDEX_WIDTH;

//! These kernels are arch-generic; the only runtime requirement is any CUDA device.
bool hasCudaDevice()
{
    int32_t deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
}

//! cos/sin table [maxPositions, 64] matching initializeNormalRopeCosSin with rotaryDim = 64
//! (cos in columns [0, 32), sin in [32, 64)); theta = 1e7 is the qwen4_exp indexer rope base.
std::vector<float> makeCosSinTable(int32_t maxPositions)
{
    std::vector<float> table(static_cast<size_t>(maxPositions) * qsa_ref::kRotaryDim);
    for (int32_t pos = 0; pos < maxPositions; ++pos)
    {
        for (int32_t i = 0; i < qsa_ref::kRotaryDim / 2; ++i)
        {
            double const angle = pos / std::pow(1e7, 2.0 * i / qsa_ref::kRotaryDim);
            table[static_cast<int64_t>(pos) * qsa_ref::kRotaryDim + i] = static_cast<float>(std::cos(angle));
            table[static_cast<int64_t>(pos) * qsa_ref::kRotaryDim + qsa_ref::kRotaryDim / 2 + i]
                = static_cast<float>(std::sin(angle));
        }
    }
    return table;
}

//! Host-side random inputs (kept as both device halves and their half-rounded FP32 mirror)
//! plus the device tensors every stage consumes.
struct QsaIndexerHarness
{
    int32_t batch;
    int32_t seqLen;
    int32_t numBlocks;
    float rmsEps{1e-6f};
    std::vector<int32_t> lens;
    std::vector<half> indexQkH;
    std::vector<float> indexQkF; //!< half-rounded values the GPU sees
    std::vector<float> cosSinF;
    std::vector<float> wQF;
    std::vector<float> wKF;
    rt::Tensor dIndexQk;
    rt::Tensor dCosSin;
    rt::Tensor dLens;
    rt::Tensor dWQ;
    rt::Tensor dWK;

    QsaIndexerHarness(int32_t batch_, int32_t seqLen_, std::vector<int32_t> lens_, uint32_t seed)
        : batch(batch_)
        , seqLen(seqLen_)
        , numBlocks(qsa_ref::ceilDiv(seqLen_, qsa_ref::kRatio))
        , lens(std::move(lens_))
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> activationDist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> gammaDist(-0.5f, 0.5f);

        indexQkH.resize(static_cast<size_t>(batch) * seqLen * qsa_ref::kIndexQkWidth);
        indexQkF.resize(indexQkH.size());
        for (size_t i = 0; i < indexQkH.size(); ++i)
        {
            indexQkH[i] = __float2half(activationDist(rng));
            indexQkF[i] = __half2float(indexQkH[i]);
        }

        std::vector<half> wQH(qsa_ref::kHeadDim);
        std::vector<half> wKH(qsa_ref::kHeadDim);
        wQF.resize(qsa_ref::kHeadDim);
        wKF.resize(qsa_ref::kHeadDim);
        for (int32_t d = 0; d < qsa_ref::kHeadDim; ++d)
        {
            wQH[d] = __float2half(gammaDist(rng));
            wKH[d] = __float2half(gammaDist(rng));
            wQF[d] = __half2float(wQH[d]);
            wKF[d] = __half2float(wKH[d]);
        }

        cosSinF = makeCosSinTable(seqLen);

        dIndexQk = rt::Tensor(
            {batch, seqLen, qsa_ref::kIndexQkWidth}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "indexQk");
        dCosSin = rt::Tensor({seqLen, qsa_ref::kRotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "cosSin");
        dLens = rt::Tensor({batch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "contextLengths");
        dWQ = rt::Tensor({qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "wQ");
        dWK = rt::Tensor({qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "wK");
        uploadIndexQk(indexQkH);
        CUDA_CHECK(
            cudaMemcpy(dCosSin.rawPointer(), cosSinF.data(), cosSinF.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dLens.rawPointer(), lens.data(), lens.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dWQ.rawPointer(), wQH.data(), wQH.size() * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dWK.rawPointer(), wKH.data(), wKH.size() * sizeof(half), cudaMemcpyHostToDevice));
    }

    void uploadIndexQk(std::vector<half> const& values)
    {
        CUDA_CHECK(
            cudaMemcpy(dIndexQk.rawPointer(), values.data(), values.size() * sizeof(half), cudaMemcpyHostToDevice));
    }
};

//! Full pipeline on the harness inputs; returns host outIdx [batch * seqLen, kWidth].
std::vector<int32_t> runPrefill(QsaIndexerHarness const& h)
{
    size_t const workspaceBytes = kernel::getQsaIndexerWorkspaceSize(h.batch, h.seqLen);
    rt::Tensor workspace(
        {static_cast<int64_t>(workspaceBytes)}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "workspace");
    rt::Tensor outIdx(
        {static_cast<int64_t>(h.batch) * h.seqLen, kWidth}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "outIdx");
    kernel::runQsaIndexerPrefill<half>(outIdx.dataPointer<int32_t>(), h.dIndexQk.dataPointer<half>(),
        h.dCosSin.dataPointer<float>(), h.dLens.dataPointer<int32_t>(), h.dWQ.dataPointer<half>(),
        h.dWK.dataPointer<half>(), h.rmsEps, workspace.rawPointer(), workspaceBytes, h.batch, h.seqLen,
        /* poolState = */ nullptr, /* stream = */ nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    return copyDeviceToHost<int32_t>(outIdx);
}

void verifyPaddedRow(int32_t const* row)
{
    for (int32_t j = 0; j < kWidth; ++j)
    {
        ASSERT_EQ(row[j], -1) << "padded row slot " << j;
    }
}

//! Dense-causal invariant: nVis <= 512 rows must select EXACTLY the causal prefix {0..t}
//! (as a set; the output order is unspecified). Needs no reference.
void verifyDenseRow(int32_t const* row, int32_t t)
{
    int32_t const count = t + 1;
    std::vector<char> seen(count, 0);
    for (int32_t j = 0; j < count; ++j)
    {
        int32_t const v = row[j];
        ASSERT_GE(v, 0) << "slot " << j;
        ASSERT_LE(v, t) << "slot " << j;
        ASSERT_EQ(seen[v], 0) << "duplicate token " << v;
        seen[v] = 1;
    }
    for (int32_t j = count; j < kWidth; ++j)
    {
        ASSERT_EQ(row[j], -1) << "slot " << j;
    }
}

//! Structural checks for a sparse row (nVis > 512): exactly 512 complete blocks + the causal
//! tail, all distinct and in range, -1 padded. Returns the selected block ids.
void verifySparseRowStructure(int32_t const* row, int32_t t, std::vector<int32_t>* selectedBlocks)
{
    int32_t const numVisible = (t + 1) / qsa_ref::kRatio;
    int32_t const tailCount = (t + 1) % qsa_ref::kRatio;
    int32_t const nonPadCount = qsa_ref::kTopk * qsa_ref::kRatio + tailCount;
    ASSERT_GT(numVisible, qsa_ref::kTopk);

    std::vector<char> seen(t + 1, 0);
    std::vector<int32_t> blockCount(numVisible, 0);
    for (int32_t j = 0; j < nonPadCount; ++j)
    {
        int32_t const v = row[j];
        ASSERT_GE(v, 0) << "slot " << j;
        ASSERT_LE(v, t) << "slot " << j;
        ASSERT_EQ(seen[v], 0) << "duplicate token " << v;
        seen[v] = 1;
        if (v < numVisible * qsa_ref::kRatio)
        {
            ++blockCount[v / qsa_ref::kRatio];
        }
    }
    for (int32_t j = nonPadCount; j < kWidth; ++j)
    {
        ASSERT_EQ(row[j], -1) << "slot " << j;
    }
    for (int32_t j = 0; j < tailCount; ++j)
    {
        ASSERT_EQ(seen[numVisible * qsa_ref::kRatio + j], 1) << "missing tail token";
    }
    selectedBlocks->clear();
    for (int32_t g = 0; g < numVisible; ++g)
    {
        ASSERT_TRUE(blockCount[g] == 0 || blockCount[g] == qsa_ref::kRatio)
            << "block " << g << " partially selected (" << blockCount[g] << " tokens)";
        if (blockCount[g] == qsa_ref::kRatio)
        {
            selectedBlocks->push_back(g);
        }
    }
    ASSERT_EQ(static_cast<int32_t>(selectedBlocks->size()), qsa_ref::kTopk);
}

//! Index-SET comparison against the reference top-512 with a tie band: blocks in the
//! symmetric difference are only allowed when their logit is within
//! 1e-3 * max(1, |kth logit|) of the kth (smallest selected) reference logit.
void verifyTieBandSelection(std::vector<int32_t> const& gpuBlocks, std::vector<double> const& logits)
{
    std::vector<int32_t> const refIds = qsa_ref::refSortedIds(logits);
    double const kthLogit = logits[refIds[qsa_ref::kTopk - 1]];
    double const band = 1e-3 * std::max(1.0, std::abs(kthLogit));

    std::vector<char> inRef(logits.size(), 0);
    for (int32_t i = 0; i < qsa_ref::kTopk; ++i)
    {
        inRef[refIds[i]] = 1;
    }
    std::vector<char> inGpu(logits.size(), 0);
    for (int32_t g : gpuBlocks)
    {
        inGpu[g] = 1;
    }
    for (size_t g = 0; g < logits.size(); ++g)
    {
        if (inRef[g] != inGpu[g])
        {
            ASSERT_LE(std::abs(logits[g] - kthLogit), band)
                << "block " << g << " outside tie band: logit " << logits[g] << " vs kth " << kthLogit;
        }
    }
}

} // namespace

TEST(QsaIndexerKernels, QPrepMatchesReference)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kBatch = 2;
    constexpr int32_t kSeqLen = 67; // deliberately not a multiple of 4
    QsaIndexerHarness h(kBatch, kSeqLen, {50, 67}, /* seed = */ 123);

    rt::Tensor dQNormed({kBatch, kSeqLen, qsa_ref::kNumHeads, qsa_ref::kHeadDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "qNormed");
    kernel::launchQsaIndexQPrep<half>(dQNormed.dataPointer<half>(), h.dIndexQk.dataPointer<half>(),
        h.dCosSin.dataPointer<float>(), h.dLens.dataPointer<int32_t>(), h.dWQ.dataPointer<half>(), h.rmsEps, kBatch,
        kSeqLen, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> const got = copyDeviceToHost<half>(dQNormed);

    std::vector<float> const ref = qsa_ref::refQPrep(h.indexQkF, h.cosSinF, h.lens, h.wQF, h.rmsEps, kBatch, kSeqLen);
    ASSERT_EQ(got.size(), ref.size());
    for (int32_t b = 0; b < kBatch; ++b)
    {
        for (int32_t t = 0; t < kSeqLen; ++t)
        {
            for (int32_t i = 0; i < qsa_ref::kQRowWidth; ++i)
            {
                int64_t const idx = (static_cast<int64_t>(b) * kSeqLen + t) * qsa_ref::kQRowWidth + i;
                float const g = __half2float(got[idx]);
                if (t >= h.lens[b])
                {
                    ASSERT_EQ(g, 0.0f) << "padding row not zero at b=" << b << " t=" << t << " i=" << i;
                }
                else
                {
                    ASSERT_TRUE(isclose(g, ref[idx], /* rtol = */ 1e-2f, /* atol = */ 1e-3f))
                        << "b=" << b << " t=" << t << " i=" << i << " got " << g << " want " << ref[idx];
                }
            }
        }
    }
}

TEST(QsaIndexerKernels, KCompressMatchesReference)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kBatch = 2;
    constexpr int32_t kSeqLen = 67;
    QsaIndexerHarness h(kBatch, kSeqLen, {50, 67}, /* seed = */ 321);
    int32_t const numBlocks = h.numBlocks;

    rt::Tensor dKbar({kBatch, numBlocks, qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "kbar");
    kernel::launchQsaIndexKCompress<half>(dKbar.dataPointer<half>(), h.dIndexQk.dataPointer<half>(),
        h.dCosSin.dataPointer<float>(), h.dLens.dataPointer<int32_t>(), h.dWK.dataPointer<half>(), h.rmsEps, kBatch,
        kSeqLen, numBlocks, /* blockBegin = */ 0, /* blockEnd = */ numBlocks, /* pastLen = */ 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> const got = copyDeviceToHost<half>(dKbar);

    std::vector<float> const ref
        = qsa_ref::refKCompress(h.indexQkF, h.cosSinF, h.lens, h.wKF, h.rmsEps, kBatch, kSeqLen);
    ASSERT_EQ(got.size(), ref.size());
    for (int32_t b = 0; b < kBatch; ++b)
    {
        for (int32_t g = 0; g < numBlocks; ++g)
        {
            bool const valid = g * qsa_ref::kRatio + qsa_ref::kRatio - 1 < h.lens[b];
            for (int32_t d = 0; d < qsa_ref::kHeadDim; ++d)
            {
                int64_t const idx = (static_cast<int64_t>(b) * numBlocks + g) * qsa_ref::kHeadDim + d;
                float const v = __half2float(got[idx]);
                if (!valid)
                {
                    ASSERT_EQ(v, 0.0f) << "incomplete block not zero at b=" << b << " g=" << g << " d=" << d;
                }
                else
                {
                    ASSERT_TRUE(isclose(v, ref[idx], /* rtol = */ 1e-2f, /* atol = */ 1e-3f))
                        << "b=" << b << " g=" << g << " d=" << d << " got " << v << " want " << ref[idx];
                }
            }
        }
    }
}

TEST(QsaIndexerKernels, ScoresMatchReference)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kBatch = 2;
    constexpr int32_t kSeqLen = 64;
    QsaIndexerHarness h(kBatch, kSeqLen, {64, 37}, /* seed = */ 777);
    int32_t const numBlocks = h.numBlocks;
    int32_t const numRows = kBatch * kSeqLen;

    // Feed K2 with reference-computed (half-rounded) qNormed and kbar so the test isolates K2.
    std::vector<float> const qNormedF
        = qsa_ref::halfRound(qsa_ref::refQPrep(h.indexQkF, h.cosSinF, h.lens, h.wQF, h.rmsEps, kBatch, kSeqLen));
    std::vector<float> const kbarF
        = qsa_ref::halfRound(qsa_ref::refKCompress(h.indexQkF, h.cosSinF, h.lens, h.wKF, h.rmsEps, kBatch, kSeqLen));
    std::vector<half> qNormedH(qNormedF.size());
    std::vector<half> kbarH(kbarF.size());
    for (size_t i = 0; i < qNormedF.size(); ++i)
    {
        qNormedH[i] = __float2half(qNormedF[i]);
    }
    for (size_t i = 0; i < kbarF.size(); ++i)
    {
        kbarH[i] = __float2half(kbarF[i]);
    }
    rt::Tensor dQNormed({kBatch, kSeqLen, qsa_ref::kNumHeads, qsa_ref::kHeadDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "qNormed");
    rt::Tensor dKbar({kBatch, numBlocks, qsa_ref::kHeadDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "kbar");
    copyHostToDevice(dQNormed, qNormedH);
    copyHostToDevice(dKbar, kbarH);

    rt::Tensor dLogits({numRows, numBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "logits");
    kernel::launchQsaIndexScores<half>(dLogits.dataPointer<float>(), dQNormed.dataPointer<half>(),
        dKbar.dataPointer<half>(), h.dLens.dataPointer<int32_t>(), kBatch, kSeqLen, numBlocks, /* rowStart = */ 0,
        numRows, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> const got = copyDeviceToHost<float>(dLogits);

    for (int32_t b = 0; b < kBatch; ++b)
    {
        for (int32_t t = 0; t < kSeqLen; ++t)
        {
            std::vector<double> const ref = qsa_ref::refLogitsRow(qNormedF, kbarF, h.lens, kBatch, kSeqLen, b, t);
            for (int32_t g = 0; g < numBlocks; ++g)
            {
                float const v = got[(static_cast<int64_t>(b) * kSeqLen + t) * numBlocks + g];
                bool const masked = (t >= h.lens[b]) || (g >= (t + 1) / qsa_ref::kRatio);
                if (masked)
                {
                    ASSERT_EQ(v, -FLT_MAX) << "b=" << b << " t=" << t << " g=" << g;
                }
                else
                {
                    ASSERT_TRUE(isclose(v, static_cast<float>(ref[g]), /* rtol = */ 1e-3f, /* atol = */ 5e-3f))
                        << "b=" << b << " t=" << t << " g=" << g << " got " << v << " want " << ref[g];
                }
            }
        }
    }
}

TEST(QsaIndexerKernels, ExpandMatchesReferenceExactly)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    // Two geometries: a short one and one with nVis > 512 at the last row (kSel clamping).
    struct Case
    {
        int32_t batch;
        int32_t seqLen;
        std::vector<int32_t> lens;
    };
    std::vector<Case> const cases = {{2, 67, {67, 23}}, {1, 2052, {2052}}};
    std::mt19937 rng(2026);

    for (Case const& c : cases)
    {
        int32_t const numBlocks = qsa_ref::ceilDiv(c.seqLen, qsa_ref::kRatio);
        int32_t const numRows = c.batch * c.seqLen;

        // K4 only consumes sorted block ids, so any per-row permutation of [0, numBlocks)
        // exercises it exactly; expansion must match the reference integer for integer.
        std::vector<int32_t> sortedIds(static_cast<size_t>(numRows) * numBlocks);
        for (int32_t r = 0; r < numRows; ++r)
        {
            int32_t* row = sortedIds.data() + static_cast<int64_t>(r) * numBlocks;
            std::iota(row, row + numBlocks, 0);
            std::shuffle(row, row + numBlocks, rng);
        }

        rt::Tensor dSortedIds({numRows, numBlocks}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "sortedIds");
        rt::Tensor dLens({c.batch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "lens");
        rt::Tensor dOutIdx({numRows, kWidth}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "outIdx");
        copyHostToDevice(dSortedIds, sortedIds);
        copyHostToDevice(dLens, c.lens);

        kernel::launchQsaIndexExpand(dOutIdx.dataPointer<int32_t>(), dSortedIds.dataPointer<int32_t>(),
            dLens.dataPointer<int32_t>(), c.batch, c.seqLen, numBlocks, /* rowStart = */ 0, numRows, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int32_t> const got = copyDeviceToHost<int32_t>(dOutIdx);

        for (int32_t b = 0; b < c.batch; ++b)
        {
            for (int32_t t = 0; t < c.seqLen; ++t)
            {
                int64_t const r = static_cast<int64_t>(b) * c.seqLen + t;
                std::vector<int32_t> const rowIds(
                    sortedIds.begin() + r * numBlocks, sortedIds.begin() + (r + 1) * numBlocks);
                std::vector<int32_t> const ref = qsa_ref::refExpandRow(rowIds, t, c.lens[b]);
                for (int32_t j = 0; j < kWidth; ++j)
                {
                    ASSERT_EQ(got[r * kWidth + j], ref[j]) << "b=" << b << " t=" << t << " slot " << j;
                }
            }
        }
    }
}

TEST(QsaIndexerPrefill, MixedLengthsE2E)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kBatch = 2;
    constexpr int32_t kSeqLen = 2176;
    // Covers L in {1,2,3,4,5,7,8,2047,2048,2051,2052} with mixed lengths per run. L = 2052
    // is the only sparse case (row t = 2051 sees 513 blocks and must drop exactly one).
    std::vector<std::pair<int32_t, int32_t>> const lengthPairs
        = {{1, 2}, {3, 4}, {5, 7}, {8, 2047}, {2048, 2051}, {2052, 1}};

    uint32_t seed = 42;
    for (auto const& [len0, len1] : lengthPairs)
    {
        SCOPED_TRACE("lens={" + std::to_string(len0) + "," + std::to_string(len1) + "}");
        QsaIndexerHarness h(kBatch, kSeqLen, {len0, len1}, seed++);
        std::vector<int32_t> const out = runPrefill(h);

        // Reference logits are only needed for sparse rows (nVis > 512 <=> t >= 2051).
        bool const needReference = std::max(len0, len1) > qsa_ref::kTopk * qsa_ref::kRatio + qsa_ref::kRatio - 1;
        std::vector<float> qNormedF;
        std::vector<float> kbarF;
        if (needReference)
        {
            qNormedF = qsa_ref::halfRound(
                qsa_ref::refQPrep(h.indexQkF, h.cosSinF, h.lens, h.wQF, h.rmsEps, kBatch, kSeqLen));
            kbarF = qsa_ref::halfRound(
                qsa_ref::refKCompress(h.indexQkF, h.cosSinF, h.lens, h.wKF, h.rmsEps, kBatch, kSeqLen));
        }

        for (int32_t b = 0; b < kBatch; ++b)
        {
            for (int32_t t = 0; t < kSeqLen; ++t)
            {
                SCOPED_TRACE("b=" + std::to_string(b) + " t=" + std::to_string(t));
                int32_t const* row = out.data() + (static_cast<int64_t>(b) * kSeqLen + t) * kWidth;
                if (t >= h.lens[b])
                {
                    verifyPaddedRow(row);
                }
                else if ((t + 1) / qsa_ref::kRatio <= qsa_ref::kTopk)
                {
                    verifyDenseRow(row, t);
                }
                else
                {
                    std::vector<int32_t> gpuBlocks;
                    verifySparseRowStructure(row, t, &gpuBlocks);
                    if (::testing::Test::HasFatalFailure())
                    {
                        return;
                    }
                    std::vector<double> const logits
                        = qsa_ref::refLogitsRow(qNormedF, kbarF, h.lens, kBatch, kSeqLen, b, t);
                    verifyTieBandSelection(gpuBlocks, logits);
                }
                if (::testing::Test::HasFatalFailure())
                {
                    return;
                }
            }
        }
    }
}

TEST(QsaIndexerPrefill, NonMultipleOfFourSeqLen)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kBatch = 2;
    constexpr int32_t kSeqLen = 133; // S % 4 == 1
    QsaIndexerHarness h(kBatch, kSeqLen, {133, 66}, /* seed = */ 9);
    std::vector<int32_t> const out = runPrefill(h);

    for (int32_t b = 0; b < kBatch; ++b)
    {
        for (int32_t t = 0; t < kSeqLen; ++t)
        {
            SCOPED_TRACE("b=" + std::to_string(b) + " t=" + std::to_string(t));
            int32_t const* row = out.data() + (static_cast<int64_t>(b) * kSeqLen + t) * kWidth;
            if (t >= h.lens[b])
            {
                verifyPaddedRow(row);
            }
            else
            {
                verifyDenseRow(row, t); // nVis <= 34 << 512: exact causal prefix
            }
            if (::testing::Test::HasFatalFailure())
            {
                return;
            }
        }
    }
}

TEST(QsaIndexerPrefill, PaddingRowsAreNaNSafe)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    constexpr int32_t kBatch = 2;
    constexpr int32_t kSeqLen = 64;
    QsaIndexerHarness h(kBatch, kSeqLen, {37, 5}, /* seed = */ 555);
    std::vector<int32_t> const clean = runPrefill(h);

    // Poison every padding row of indexQk with NaN: the pipeline must never read them.
    std::vector<half> poisoned = h.indexQkH;
    half const nan = __float2half(std::numeric_limits<float>::quiet_NaN());
    for (int32_t b = 0; b < kBatch; ++b)
    {
        for (int32_t t = h.lens[b]; t < kSeqLen; ++t)
        {
            int64_t const offset = (static_cast<int64_t>(b) * kSeqLen + t) * qsa_ref::kIndexQkWidth;
            std::fill_n(poisoned.begin() + offset, qsa_ref::kIndexQkWidth, nan);
        }
    }
    h.uploadIndexQk(poisoned);
    std::vector<int32_t> const dirty = runPrefill(h);

    for (int32_t b = 0; b < kBatch; ++b)
    {
        for (int32_t t = 0; t < kSeqLen; ++t)
        {
            int64_t const base = (static_cast<int64_t>(b) * kSeqLen + t) * kWidth;
            if (t >= h.lens[b])
            {
                verifyPaddedRow(dirty.data() + base);
                if (::testing::Test::HasFatalFailure())
                {
                    return;
                }
            }
            else
            {
                for (int32_t j = 0; j < kWidth; ++j)
                {
                    ASSERT_EQ(dirty[base + j], clean[base + j]) << "b=" << b << " t=" << t << " slot " << j;
                }
            }
        }
    }
}

TEST(QsaIndexerPrefill, MultiChunkDeterministicRuns)
{
    if (!hasCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device available";
    }
    // B * S = 8192 rows with MB = 2048 blocks: the 32 MiB sort-keys cap yields a 4096-row
    // chunk, so this run exercises the chunk loop twice.
    constexpr int32_t kBatch = 1;
    constexpr int32_t kSeqLen = 8192;
    QsaIndexerHarness h(kBatch, kSeqLen, {kSeqLen}, /* seed = */ 31337);

    std::vector<int32_t> const first = runPrefill(h);
    std::vector<int32_t> const second = runPrefill(h);
    ASSERT_TRUE(first == second) << "outIdx must be bitwise identical across identical runs";

    for (int32_t t = 0; t < kSeqLen; ++t)
    {
        SCOPED_TRACE("t=" + std::to_string(t));
        int32_t const* row = first.data() + static_cast<int64_t>(t) * kWidth;
        if ((t + 1) / qsa_ref::kRatio <= qsa_ref::kTopk)
        {
            verifyDenseRow(row, t);
        }
        else
        {
            std::vector<int32_t> gpuBlocks;
            verifySparseRowStructure(row, t, &gpuBlocks);
        }
        if (::testing::Test::HasFatalFailure())
        {
            return;
        }
    }
}
