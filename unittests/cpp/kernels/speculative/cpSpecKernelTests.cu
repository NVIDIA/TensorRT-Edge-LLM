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

#include "common/cudaUtils.h"
#include "kernels/speculative/cpSpecKernels.h"
#include "kernels/speculative/dsparkKernels.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using namespace nvinfer1;

namespace
{

constexpr int32_t kCodebookSize = 2048;
constexpr int32_t kCpTopK = 50;
constexpr float kCpTopP = 0.8F;
constexpr float kCpTemperature = 0.9F;

std::vector<float> randomLogits(int32_t rows, int32_t vocabSize, uint32_t seed, float scale = 4.0F)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, scale);
    std::vector<float> out(static_cast<size_t>(rows) * vocabSize);
    for (auto& value : out)
    {
        value = dist(rng);
    }
    return out;
}

//! Both kernels are run on the same logits; the CP path must reproduce the
//! DSpark reference distribution it specialises.
void expectSameDistribution(std::vector<float> const& logitsHost, int32_t rows, int32_t vocabSize, float temperature,
    int32_t topK, float topP, float atol)
{
    cudaStream_t stream = nullptr;
    auto logits = rt::Tensor({rows, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto reference = rt::Tensor({rows, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto actual = rt::Tensor({rows, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    copyHostToDevice<float>(logits, logitsHost);

    dsparkLogitsToProbabilities(logits, reference, rows, vocabSize, temperature, topK, topP, stream);
    cpSpecTopKTopPProbs(logits, actual, rows, vocabSize, temperature, topK, topP, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const refHost = copyDeviceToHost<float>(reference);
    auto const actHost = copyDeviceToHost<float>(actual);
    ASSERT_EQ(refHost.size(), actHost.size());

    for (int32_t row = 0; row < rows; ++row)
    {
        double totalVariation = 0.0;
        double refSum = 0.0;
        double actSum = 0.0;
        for (int32_t v = 0; v < vocabSize; ++v)
        {
            size_t const idx = static_cast<size_t>(row) * vocabSize + v;
            EXPECT_NEAR(actHost[idx], refHost[idx], atol) << "row=" << row << " token=" << v;
            totalVariation += std::abs(static_cast<double>(actHost[idx]) - refHost[idx]);
            refSum += refHost[idx];
            actSum += actHost[idx];
        }
        EXPECT_NEAR(refSum, 1.0, 1e-4) << "row=" << row;
        EXPECT_NEAR(actSum, 1.0, 1e-4) << "row=" << row;
        EXPECT_LT(0.5 * totalVariation, 1e-4) << "row=" << row;
    }
}

} // namespace

TEST(CpSpecKernels, TopKTopPMatchesDSparkOnCodebookSizedRows)
{
    constexpr int32_t rows = 8;
    expectSameDistribution(randomLogits(rows, kCodebookSize, 1234U), rows, kCodebookSize, kCpTemperature, kCpTopK,
        kCpTopP, /*atol=*/1e-5F);
}

TEST(CpSpecKernels, TopKTopPMatchesDSparkWithoutTopP)
{
    constexpr int32_t rows = 4;
    expectSameDistribution(randomLogits(rows, kCodebookSize, 99U), rows, kCodebookSize, kCpTemperature, kCpTopK,
        /*topP=*/1.0F, /*atol=*/1e-5F);
}

TEST(CpSpecKernels, TopKTopPMatchesDSparkOnPeakedRows)
{
    // Peaked rows are the CP's normal regime and drive the nucleus down to a
    // handful of tokens, which exercises the top-p truncation boundary.
    constexpr int32_t rows = 4;
    expectSameDistribution(randomLogits(rows, kCodebookSize, 7U, /*scale=*/12.0F), rows, kCodebookSize, kCpTemperature,
        kCpTopK, kCpTopP, /*atol=*/1e-5F);
}

TEST(CpSpecKernels, TopKTopPHalfLogitsMatchTheFloatPath)
{
    // The runtime feeds the lm_head GEMV output straight in as FP16; rounding the
    // reference input to half first isolates the dtype path from precision loss.
    cudaStream_t stream = nullptr;
    constexpr int32_t rows = 4;
    auto const raw = randomLogits(rows, kCodebookSize, 31337U);
    std::vector<float> rounded(raw.size());
    std::vector<__half> halves(raw.size());
    for (size_t i = 0; i < raw.size(); ++i)
    {
        halves[i] = __float2half(raw[i]);
        rounded[i] = __half2float(halves[i]);
    }

    auto floatLogits = rt::Tensor({rows, kCodebookSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto halfLogits = rt::Tensor({rows, kCodebookSize}, rt::DeviceType::kGPU, DataType::kHALF);
    auto reference = rt::Tensor({rows, kCodebookSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto actual = rt::Tensor({rows, kCodebookSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    copyHostToDevice<float>(floatLogits, rounded);
    copyHostToDevice<__half>(halfLogits, halves);

    dsparkLogitsToProbabilities(floatLogits, reference, rows, kCodebookSize, kCpTemperature, kCpTopK, kCpTopP, stream);
    cpSpecTopKTopPProbs(halfLogits, actual, rows, kCodebookSize, kCpTemperature, kCpTopK, kCpTopP, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const refHost = copyDeviceToHost<float>(reference);
    auto const actHost = copyDeviceToHost<float>(actual);
    for (int32_t row = 0; row < rows; ++row)
    {
        double totalVariation = 0.0;
        for (int32_t v = 0; v < kCodebookSize; ++v)
        {
            size_t const idx = static_cast<size_t>(row) * kCodebookSize + v;
            totalVariation += std::abs(static_cast<double>(actHost[idx]) - refHost[idx]);
        }
        EXPECT_LT(0.5 * totalVariation, 1e-4) << "row=" << row;
    }
}

TEST(CpSpecKernels, TopKTopPGreedyTemperatureCollapsesToArgmax)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t rows = 1;
    constexpr int32_t vocabSize = 8;

    auto logits = rt::Tensor({rows, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto probabilities = rt::Tensor({rows, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    copyHostToDevice<float>(logits, {1.0F, 5.0F, 2.0F, 0.0F, 4.0F, 3.0F, -1.0F, 2.5F});

    cpSpecTopKTopPProbs(logits, probabilities, rows, vocabSize, /*temperature=*/0.0F, kCpTopK, kCpTopP, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const host = copyDeviceToHost<float>(probabilities);
    EXPECT_NEAR(host[1], 1.0F, 1e-6F);
    for (int32_t v = 0; v < vocabSize; ++v)
    {
        if (v != 1)
        {
            EXPECT_NEAR(host[v], 0.0F, 1e-6F) << "token=" << v;
        }
    }
}

TEST(CpSpecKernels, SampleRowsDrawsFromTheGivenDistribution)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t rows = 4096;
    constexpr int32_t vocabSize = 4;

    auto probabilities = rt::Tensor({rows, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto uniforms = rt::Tensor({rows}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto tokenIds = rt::Tensor({rows}, rt::DeviceType::kGPU, DataType::kINT32);

    std::vector<float> const target{0.5F, 0.25F, 0.125F, 0.125F};
    std::vector<float> probsHost(static_cast<size_t>(rows) * vocabSize);
    for (int32_t r = 0; r < rows; ++r)
    {
        std::copy(target.begin(), target.end(), probsHost.begin() + static_cast<size_t>(r) * vocabSize);
    }
    copyHostToDevice<float>(probabilities, probsHost);
    dsparkFillUniforms(uniforms, rows, /*philoxSeed=*/0xC0FFEEULL, /*philoxOffset=*/0ULL, stream);
    cpSpecSampleRows(probabilities, uniforms.dataPointer<float>(), tokenIds, rows, vocabSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const drawn = copyDeviceToHost<int32_t>(tokenIds);
    std::vector<int32_t> counts(vocabSize, 0);
    for (auto id : drawn)
    {
        ASSERT_GE(id, 0);
        ASSERT_LT(id, vocabSize);
        ++counts[id];
    }
    for (int32_t v = 0; v < vocabSize; ++v)
    {
        EXPECT_NEAR(static_cast<float>(counts[v]) / rows, target[v], 0.03F) << "token=" << v;
    }
}

TEST(CpSpecKernels, ProbabilisticAcceptMatchesDSparkVerdicts)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 16;
    constexpr int32_t vocabSize = 64;
    constexpr int32_t proposalLen = 3;
    constexpr int32_t verifyLen = proposalLen + 1;
    constexpr int32_t uniformStride = 2 * proposalLen + 1;

    auto targetProbs = rt::Tensor({batchSize, verifyLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftProbs = rt::Tensor({batchSize, proposalLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto draftIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto proposalLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptUniforms = rt::Tensor({batchSize, uniformStride}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto refIds = rt::Tensor({batchSize, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto refLen = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto actIds = rt::Tensor({batchSize, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto actLen = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

    std::mt19937 rng(4242U);
    std::uniform_real_distribution<float> dist(0.05F, 1.0F);
    auto makeRows = [&](int32_t rows) {
        std::vector<float> out(static_cast<size_t>(rows) * vocabSize);
        for (int32_t r = 0; r < rows; ++r)
        {
            float sum = 0.0F;
            for (int32_t v = 0; v < vocabSize; ++v)
            {
                float const p = dist(rng);
                out[static_cast<size_t>(r) * vocabSize + v] = p;
                sum += p;
            }
            for (int32_t v = 0; v < vocabSize; ++v)
            {
                out[static_cast<size_t>(r) * vocabSize + v] /= sum;
            }
        }
        return out;
    };
    copyHostToDevice<float>(targetProbs, makeRows(batchSize * verifyLen));
    copyHostToDevice<float>(draftProbs, makeRows(batchSize * proposalLen));

    std::vector<int32_t> ids(static_cast<size_t>(batchSize) * proposalLen);
    std::uniform_int_distribution<int32_t> tokenDist(0, vocabSize - 1);
    for (auto& id : ids)
    {
        id = tokenDist(rng);
    }
    copyHostToDevice<int32_t>(draftIds, ids);
    copyHostToDevice<int32_t>(proposalLengths, std::vector<int32_t>(batchSize, proposalLen));
    dsparkFillUniforms(acceptUniforms, batchSize * uniformStride, /*philoxSeed=*/0x51ED270BULL, 0ULL, stream);

    dsparkProbabilisticAccept(targetProbs, draftProbs, draftIds, proposalLengths, acceptUniforms, refIds, refLen,
        batchSize, proposalLen, proposalLen, vocabSize, stream);
    cpSpecProbabilisticAccept(targetProbs, draftProbs, draftIds, proposalLengths, acceptUniforms.dataPointer<float>(),
        actIds, actLen, batchSize, proposalLen, proposalLen, vocabSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const refLenHost = copyDeviceToHost<int32_t>(refLen);
    auto const actLenHost = copyDeviceToHost<int32_t>(actLen);
    auto const refIdsHost = copyDeviceToHost<int32_t>(refIds);
    auto const actIdsHost = copyDeviceToHost<int32_t>(actIds);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        EXPECT_EQ(actLenHost[b], refLenHost[b]) << "batch=" << b;
        for (int32_t i = 0; i < refLenHost[b]; ++i)
        {
            EXPECT_EQ(actIdsHost[b * verifyLen + i], refIdsHost[b * verifyLen + i]) << "batch=" << b << " pos=" << i;
        }
    }
}

TEST(CpSpecKernels, GroupedHeadLinearMatchesAnFp32Reference)
{
    // Each verify position is scored by a different lm_head, which the engine's single
    // lm_head_idx gather cannot express, so the runtime applies the heads itself. Check
    // the fused per-row GEMV against an FP32 host reference.
    cudaStream_t stream = nullptr;
    constexpr int32_t numHeads = 4;
    constexpr int32_t hiddenDim = 1024;
    constexpr int32_t outputDim = 512;
    constexpr int32_t rows = 3;
    std::vector<int32_t> const hiddenSel{0, 1, 2};
    std::vector<int32_t> const headSel{1, 3, 2};

    std::mt19937 rng(20260818U);
    std::normal_distribution<float> dist(0.0F, 0.05F);
    auto fill = [&](size_t n) {
        std::vector<__half> v(n);
        for (auto& x : v)
        {
            x = __float2half(dist(rng));
        }
        return v;
    };
    auto const hiddensHost = fill(static_cast<size_t>(rows) * hiddenDim);
    auto const headsHost = fill(static_cast<size_t>(numHeads) * outputDim * hiddenDim);

    auto hiddens = rt::Tensor({rows, hiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto heads = rt::Tensor({numHeads, outputDim, hiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto rowHidden = rt::Tensor({rows}, rt::DeviceType::kGPU, DataType::kINT32);
    auto rowHead = rt::Tensor({rows}, rt::DeviceType::kGPU, DataType::kINT32);
    auto fused = rt::Tensor({rows, outputDim}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice<__half>(hiddens, hiddensHost);
    copyHostToDevice<__half>(heads, headsHost);
    copyHostToDevice<int32_t>(rowHidden, hiddenSel);
    copyHostToDevice<int32_t>(rowHead, headSel);

    invokeGroupedHeadLinear(static_cast<__half const*>(hiddens.rawPointer()), heads, rowHidden.dataPointer<int32_t>(),
        rowHead.dataPointer<int32_t>(), rows, hiddenDim, outputDim, fused, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto const fusedHost = copyDeviceToHost<__half>(fused);

    for (int32_t r = 0; r < rows; ++r)
    {
        int32_t refArgmax = 0;
        int32_t gpuArgmax = 0;
        float refBest = -1e30F;
        float gpuBest = -1e30F;
        for (int32_t o = 0; o < outputDim; ++o)
        {
            double acc = 0.0;
            for (int32_t d = 0; d < hiddenDim; ++d)
            {
                acc += static_cast<double>(__half2float(hiddensHost[hiddenSel[r] * hiddenDim + d]))
                    * __half2float(headsHost[(static_cast<size_t>(headSel[r]) * outputDim + o) * hiddenDim + d]);
            }
            float const gpu = __half2float(fusedHost[static_cast<size_t>(r) * outputDim + o]);
            EXPECT_NEAR(gpu, static_cast<float>(acc), 5e-3F) << "row=" << r << " out=" << o;
            if (acc > refBest)
            {
                refBest = static_cast<float>(acc);
                refArgmax = o;
            }
            if (gpu > gpuBest)
            {
                gpuBest = gpu;
                gpuArgmax = o;
            }
        }
        EXPECT_EQ(gpuArgmax, refArgmax) << "row=" << r;
    }
}

TEST(CpSpecKernels, GatherCodecEmbedRowsHandlesTheProjectedTalkerWidth)
{
    // Checkpoints whose Talker and CodePredictor hidden sizes differ gather at the Talker
    // width into a staging buffer and project afterwards, so the gather must be width- and
    // table-agnostic rather than assuming the CodePredictor hidden size.
    cudaStream_t stream = nullptr;
    constexpr int32_t numTables = 4;
    constexpr int32_t codebook = 64;
    constexpr int32_t hiddenDim = 1280;
    constexpr int32_t batch = 2;
    constexpr int32_t rows = 3;
    constexpr int32_t codeStride = 8;

    std::mt19937 rng(4711U);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<std::vector<__half>> tablesHost(numTables);
    std::vector<rt::Tensor> tables;
    std::vector<void const*> tablePtrsHost(numTables);
    for (int32_t t = 0; t < numTables; ++t)
    {
        tablesHost[t].resize(static_cast<size_t>(codebook) * hiddenDim);
        for (auto& x : tablesHost[t])
        {
            x = __float2half(dist(rng));
        }
        tables.emplace_back(rt::Coords{codebook, hiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
        copyHostToDevice<__half>(tables.back(), tablesHost[t]);
        tablePtrsHost[t] = tables.back().rawPointer();
    }
    auto tablePtrs
        = rt::Tensor({static_cast<int64_t>(numTables * sizeof(void*))}, rt::DeviceType::kGPU, DataType::kINT8);
    CUDA_CHECK(
        cudaMemcpy(tablePtrs.rawPointer(), tablePtrsHost.data(), numTables * sizeof(void*), cudaMemcpyHostToDevice));

    std::vector<int32_t> codesHost(static_cast<size_t>(batch) * codeStride, 0);
    std::vector<int32_t> tableIdxHost(static_cast<size_t>(batch) * rows, 0);
    std::uniform_int_distribution<int32_t> codeDist(0, codebook - 1);
    std::uniform_int_distribution<int32_t> tblDist(0, numTables - 1);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t i = 0; i < rows; ++i)
        {
            codesHost[static_cast<size_t>(b) * codeStride + i] = codeDist(rng);
            tableIdxHost[static_cast<size_t>(b) * rows + i] = tblDist(rng);
        }
    }
    auto codeIds = rt::Tensor({batch, codeStride}, rt::DeviceType::kGPU, DataType::kINT32);
    auto tableIdx = rt::Tensor({batch, rows}, rt::DeviceType::kGPU, DataType::kINT32);
    auto output = rt::Tensor({batch, rows, hiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice<int32_t>(codeIds, codesHost);
    copyHostToDevice<int32_t>(tableIdx, tableIdxHost);

    invokeGatherCodecEmbedRows(codeIds, static_cast<__half const* const*>(tablePtrs.rawPointer()),
        tableIdx.dataPointer<int32_t>(), batch, rows, codeStride, hiddenDim, output, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const got = copyDeviceToHost<__half>(output);
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t i = 0; i < rows; ++i)
        {
            int32_t const tbl = tableIdxHost[static_cast<size_t>(b) * rows + i];
            int32_t const code = codesHost[static_cast<size_t>(b) * codeStride + i];
            for (int32_t d = 0; d < hiddenDim; ++d)
            {
                float const expected = __half2float(tablesHost[tbl][static_cast<size_t>(code) * hiddenDim + d]);
                float const actual = __half2float(got[(static_cast<size_t>(b) * rows + i) * hiddenDim + d]);
                ASSERT_EQ(actual, expected) << "batch=" << b << " row=" << i << " dim=" << d;
            }
        }
    }
}

//! One CodePredictor speculative round, wired the way the runtime wires it:
//! stacked lm_heads produce the draft logits for the depths ahead of the committed
//! one, both sides are filtered into dense rows, the draft is sampled, and the
//! verifier decides how much of the frame the round commits.
namespace
{

struct MockRound
{
    std::vector<int32_t> acceptLength;
    std::vector<int32_t> committed;
    std::vector<int32_t> draftIds;
};

//! @param targetIsDraft feeds the verifier the draft distribution as the target, which is the
//!        p == q case where the accept test min(1, p/q) can never reject.
MockRound runMockRound(int32_t batchSize, int32_t verifyLen, int32_t vocabSize, bool targetIsDraft, uint32_t seed)
{
    cudaStream_t stream = nullptr;
    int32_t const proposalLen = verifyLen - 1;
    int32_t const uniformStride = 2 * proposalLen + 1;

    auto draftLogits = rt::Tensor({batchSize * proposalLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto targetLogits = rt::Tensor({batchSize * verifyLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    copyHostToDevice<float>(draftLogits, randomLogits(batchSize * proposalLen, vocabSize, seed));
    copyHostToDevice<float>(targetLogits, randomLogits(batchSize * verifyLen, vocabSize, seed + 1U));

    auto draftProbs = rt::Tensor({batchSize, proposalLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto targetProbs = rt::Tensor({batchSize, verifyLen, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    cpSpecTopKTopPProbs(
        draftLogits, draftProbs, batchSize * proposalLen, vocabSize, kCpTemperature, kCpTopK, kCpTopP, stream);
    cpSpecTopKTopPProbs(
        targetLogits, targetProbs, batchSize * verifyLen, vocabSize, kCpTemperature, kCpTopK, kCpTopP, stream);

    auto draftIds = rt::Tensor({batchSize, proposalLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto sampleUniforms = rt::Tensor({batchSize * proposalLen}, rt::DeviceType::kGPU, DataType::kFLOAT);
    dsparkFillUniforms(sampleUniforms, batchSize * proposalLen, /*philoxSeed=*/0x2718281ULL, seed, stream);
    cpSpecSampleRows(
        draftProbs, sampleUniforms.dataPointer<float>(), draftIds, batchSize * proposalLen, vocabSize, stream);

    auto proposalLengths = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptUniforms = rt::Tensor({batchSize, uniformStride}, rt::DeviceType::kGPU, DataType::kFLOAT);
    auto acceptedIds = rt::Tensor({batchSize, verifyLen}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptLength = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice<int32_t>(proposalLengths, std::vector<int32_t>(batchSize, proposalLen));
    dsparkFillUniforms(acceptUniforms, batchSize * uniformStride, /*philoxSeed=*/0x31415926ULL, seed, stream);

    // The p == q case still needs a verifyLen-shaped target, so reuse the draft rows for the
    // drafted depths; only the trailing bonus row comes from the target.
    if (targetIsDraft)
    {
        auto const draftHost = copyDeviceToHost<float>(draftProbs);
        auto targetHost = copyDeviceToHost<float>(targetProbs);
        for (int32_t b = 0; b < batchSize; ++b)
        {
            for (int32_t t = 0; t < proposalLen; ++t)
            {
                auto const src = (static_cast<size_t>(b) * proposalLen + t) * vocabSize;
                auto const dst = (static_cast<size_t>(b) * verifyLen + t) * vocabSize;
                std::copy_n(draftHost.begin() + src, vocabSize, targetHost.begin() + dst);
            }
        }
        copyHostToDevice<float>(targetProbs, targetHost);
    }

    cpSpecProbabilisticAccept(targetProbs, draftProbs, draftIds, proposalLengths, acceptUniforms.dataPointer<float>(),
        acceptedIds, acceptLength, batchSize, proposalLen, proposalLen, vocabSize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return MockRound{copyDeviceToHost<int32_t>(acceptLength), copyDeviceToHost<int32_t>(acceptedIds),
        copyDeviceToHost<int32_t>(draftIds)};
}

} // namespace

TEST(CpSpecKernels, MockRoundCommitsAtLeastOneCodeAndAtMostTheWindow)
{
    constexpr int32_t batchSize = 8;
    constexpr int32_t verifyLen = 4;
    auto const round = runMockRound(batchSize, verifyLen, kCodebookSize, /*targetIsDraft=*/false, 9001U);

    for (int32_t b = 0; b < batchSize; ++b)
    {
        // A round always commits the verified position, so the frame cannot stall even when
        // every draft is rejected.
        EXPECT_GE(round.acceptLength[b], 1) << "batch=" << b;
        EXPECT_LE(round.acceptLength[b], verifyLen) << "batch=" << b;

        // Everything before the last committed position is the draft that survived.
        for (int32_t i = 0; i + 1 < round.acceptLength[b]; ++i)
        {
            EXPECT_EQ(round.committed[b * verifyLen + i], round.draftIds[b * (verifyLen - 1) + i])
                << "batch=" << b << " pos=" << i;
            EXPECT_GE(round.committed[b * verifyLen + i], 0);
            EXPECT_LT(round.committed[b * verifyLen + i], kCodebookSize);
        }
    }
}

TEST(CpSpecKernels, MockRoundAcceptsEveryDraftWhenTargetEqualsDraft)
{
    constexpr int32_t batchSize = 8;
    constexpr int32_t verifyLen = 4;
    auto const round = runMockRound(batchSize, verifyLen, kCodebookSize, /*targetIsDraft=*/true, 4711U);

    for (int32_t b = 0; b < batchSize; ++b)
    {
        EXPECT_EQ(round.acceptLength[b], verifyLen) << "batch=" << b;
        for (int32_t i = 0; i + 1 < verifyLen; ++i)
        {
            EXPECT_EQ(round.committed[b * verifyLen + i], round.draftIds[b * (verifyLen - 1) + i])
                << "batch=" << b << " pos=" << i;
        }
    }
}
