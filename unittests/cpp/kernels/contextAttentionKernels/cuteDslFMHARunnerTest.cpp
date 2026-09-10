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

#ifdef CUTE_DSL_FMHA_BLACKWELL_ENABLED

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/cudaUtils.h"
#include "contextAttnReference.h"
#include "kernels/contextAttentionKernels/cuteDslFMHARunner.h"
#include "testUtils.h"

using namespace nvinfer1;
using namespace trt_edgellm;

namespace
{

bool isSupportedCuteDslTestSm(int32_t smVersion)
{
    return smVersion == 100 || smVersion == 101 || smVersion == 110;
}

void expectHalfOutputsClose(rt::Tensor const& actualTensor, rt::Tensor const& expectedTensor, std::string const& label)
{
    ASSERT_EQ(actualTensor.getShape().volume(), expectedTensor.getShape().volume()) << label;

    auto const actual = copyDeviceToHost<half>(actualTensor);
    auto const expected = copyDeviceToHost<half>(expectedTensor);
    auto const& shape = actualTensor.getShape();

    bool nanDetected = false;
    int64_t closeWithin1e3 = 0;
    int64_t const totalElements = static_cast<int64_t>(actual.size());

    for (int64_t idx = 0; idx < totalElements; ++idx)
    {
        float const actualValue = __half2float(actual[static_cast<size_t>(idx)]);
        float const expectedValue = __half2float(expected[static_cast<size_t>(idx)]);

        ASSERT_TRUE(isclose(actual[static_cast<size_t>(idx)], expected[static_cast<size_t>(idx)], 1e-2f, 1e-2f))
            << label << " mismatch at index=" << formatTensorIndex(shape, idx) << " flat_index=" << idx
            << " expected=" << expectedValue << " actual=" << actualValue;

        if (isclose(actual[static_cast<size_t>(idx)], expected[static_cast<size_t>(idx)], 1e-3f, 1e-3f))
        {
            ++closeWithin1e3;
        }

        nanDetected = nanDetected || std::isnan(actualValue);
    }

    float const passRate1e3 = static_cast<float>(closeWithin1e3) / static_cast<float>(totalElements);
    EXPECT_GT(passRate1e3, 0.9f) << label;
    EXPECT_FALSE(nanDetected) << label;
}

void expectHalfOutputRowsClose(rt::Tensor const& actualTensor, rt::Tensor const& expectedTensor,
    std::vector<int32_t> const& validSeqLens, std::string const& label)
{
    ASSERT_EQ(actualTensor.getShape(), expectedTensor.getShape()) << label;

    auto const actual = copyDeviceToHost<half>(actualTensor);
    auto const expected = copyDeviceToHost<half>(expectedTensor);
    auto const& shape = actualTensor.getShape();
    ASSERT_EQ(static_cast<int64_t>(validSeqLens.size()), shape[0]) << label;

    int64_t const rowElements = shape[2] * shape[3];
    int64_t closeWithin1e3 = 0;
    int64_t totalElements = 0;
    bool nanDetected = false;
    for (int64_t batchIdx = 0; batchIdx < shape[0]; ++batchIdx)
    {
        ASSERT_GE(validSeqLens[static_cast<size_t>(batchIdx)], 0) << label;
        ASSERT_LE(validSeqLens[static_cast<size_t>(batchIdx)], shape[1]) << label;
        for (int64_t row = 0; row < validSeqLens[static_cast<size_t>(batchIdx)]; ++row)
        {
            for (int64_t col = 0; col < rowElements; ++col)
            {
                int64_t const idx = (batchIdx * shape[1] + row) * rowElements + col;
                float const actualValue = __half2float(actual[static_cast<size_t>(idx)]);
                float const expectedValue = __half2float(expected[static_cast<size_t>(idx)]);
                ASSERT_TRUE(isclose(actual[static_cast<size_t>(idx)], expected[static_cast<size_t>(idx)], 1e-2F, 1e-2F))
                    << label << " mismatch at batch=" << batchIdx << " row=" << row << " col=" << col
                    << " expected=" << expectedValue << " actual=" << actualValue;
                if (isclose(actual[static_cast<size_t>(idx)], expected[static_cast<size_t>(idx)], 1e-3F, 1e-3F))
                {
                    ++closeWithin1e3;
                }
                nanDetected = nanDetected || std::isnan(actualValue);
                ++totalElements;
            }
        }
    }

    ASSERT_GT(totalElements, 0) << label;
    EXPECT_GT(static_cast<float>(closeWithin1e3) / static_cast<float>(totalElements), 0.9F) << label;
    EXPECT_FALSE(nanDetected) << label;
}

void expectFp8HalfOutputsClose(
    rt::Tensor const& actualTensor, rt::Tensor const& expectedTensor, std::string const& label)
{
    ASSERT_EQ(actualTensor.getShape(), expectedTensor.getShape()) << label;
    auto const actual = copyDeviceToHost<half>(actualTensor);
    auto const expected = copyDeviceToHost<half>(expectedTensor);

    double sumAbsError = 0.0;
    double sumSquaredActual = 0.0;
    double sumSquaredExpected = 0.0;
    double dot = 0.0;
    bool nanDetected = false;
    for (size_t idx = 0; idx < actual.size(); ++idx)
    {
        float const actualValue = __half2float(actual[idx]);
        float const expectedValue = __half2float(expected[idx]);
        sumAbsError += std::fabs(actualValue - expectedValue);
        sumSquaredActual += static_cast<double>(actualValue) * actualValue;
        sumSquaredExpected += static_cast<double>(expectedValue) * expectedValue;
        dot += static_cast<double>(actualValue) * expectedValue;
        nanDetected = nanDetected || std::isnan(actualValue);
    }

    double const meanAbsError = sumAbsError / static_cast<double>(actual.size());
    double const cosineSimilarity = dot / std::sqrt(std::max(sumSquaredActual * sumSquaredExpected, 1.0e-30));
    EXPECT_FALSE(nanDetected) << label;
    EXPECT_LT(meanAbsError, 0.05) << label;
    EXPECT_GT(cosineSimilarity, 0.99) << label;
}

void expectFp8HalfOutputRowsClose(rt::Tensor const& actualTensor, rt::Tensor const& expectedTensor,
    std::vector<int32_t> const& validSeqLens, std::string const& label)
{
    ASSERT_EQ(actualTensor.getShape(), expectedTensor.getShape()) << label;
    auto const actual = copyDeviceToHost<half>(actualTensor);
    auto const expected = copyDeviceToHost<half>(expectedTensor);
    auto const& shape = actualTensor.getShape();
    ASSERT_EQ(static_cast<int64_t>(validSeqLens.size()), shape[0]) << label;

    int64_t const rowElements = shape[2] * shape[3];
    double sumAbsError = 0.0;
    double sumSquaredActual = 0.0;
    double sumSquaredExpected = 0.0;
    double dot = 0.0;
    bool nanDetected = false;
    int64_t totalElements = 0;
    for (int64_t batchIdx = 0; batchIdx < shape[0]; ++batchIdx)
    {
        ASSERT_GE(validSeqLens[static_cast<size_t>(batchIdx)], 0) << label;
        ASSERT_LE(validSeqLens[static_cast<size_t>(batchIdx)], shape[1]) << label;
        for (int64_t row = 0; row < validSeqLens[static_cast<size_t>(batchIdx)]; ++row)
        {
            for (int64_t col = 0; col < rowElements; ++col)
            {
                int64_t const idx = (batchIdx * shape[1] + row) * rowElements + col;
                float const actualValue = __half2float(actual[static_cast<size_t>(idx)]);
                float const expectedValue = __half2float(expected[static_cast<size_t>(idx)]);
                sumAbsError += std::fabs(actualValue - expectedValue);
                sumSquaredActual += static_cast<double>(actualValue) * actualValue;
                sumSquaredExpected += static_cast<double>(expectedValue) * expectedValue;
                dot += static_cast<double>(actualValue) * expectedValue;
                nanDetected = nanDetected || std::isnan(actualValue);
                ++totalElements;
            }
        }
    }

    ASSERT_GT(totalElements, 0) << label;
    double const meanAbsError = sumAbsError / static_cast<double>(totalElements);
    double const cosineSimilarity = dot / std::sqrt(std::max(sumSquaredActual * sumSquaredExpected, 1.0e-30));
    EXPECT_FALSE(nanDetected) << label;
    EXPECT_LT(meanAbsError, 0.05) << label;
    EXPECT_GT(cosineSimilarity, 0.99) << label;
}

// Skip-softmax is approximate by design: skipped KV tiles perturb the output by
// up to the calibrated accuracy gate (0.1 max-abs, the same gate the baked-in
// lambda was calibrated against), so the dense comparator's 1e-2 tolerance
// does not apply.
void expectSkipSoftmaxOutputsClose(
    rt::Tensor const& actualTensor, rt::Tensor const& expectedTensor, std::string const& label)
{
    ASSERT_EQ(actualTensor.getShape().volume(), expectedTensor.getShape().volume()) << label;

    auto const actual = copyDeviceToHost<half>(actualTensor);
    auto const expected = copyDeviceToHost<half>(expectedTensor);
    auto const& shape = actualTensor.getShape();

    bool nanDetected = false;
    double sumAbsError = 0.0;
    int64_t const totalElements = static_cast<int64_t>(actual.size());

    for (int64_t idx = 0; idx < totalElements; ++idx)
    {
        float const actualValue = __half2float(actual[static_cast<size_t>(idx)]);
        float const expectedValue = __half2float(expected[static_cast<size_t>(idx)]);

        ASSERT_LT(std::fabs(actualValue - expectedValue), 0.1f)
            << label << " mismatch at index=" << formatTensorIndex(shape, idx) << " flat_index=" << idx
            << " expected=" << expectedValue << " actual=" << actualValue;

        sumAbsError += std::fabs(actualValue - expectedValue);
        nanDetected = nanDetected || std::isnan(actualValue);
    }

    double const meanAbsError = sumAbsError / static_cast<double>(totalElements);
    EXPECT_LT(meanAbsError, 0.01) << label;
    EXPECT_FALSE(nanDetected) << label;
}

void runViTAccuracyCase(
    std::vector<int32_t> const& cuSeqLens, int32_t numHeads, int32_t headDim, int32_t maxSeqLen, float attentionScale)
{
    int32_t const batchSize = static_cast<int32_t>(cuSeqLens.size()) - 1;
    int32_t const totalSeqLen = cuSeqLens.back();

    size_t const qkvSize = static_cast<size_t>(totalSeqLen) * numHeads * headDim;

    std::vector<half> qInput(qkvSize);
    std::vector<half> kInput(qkvSize);
    std::vector<half> vInput(qkvSize);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);

    rt::Tensor qTensor({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kTensor({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputReference({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputCuteDsl({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(qTensor, qInput);
    copyHostToDevice(kTensor, kInput);
    copyHostToDevice(vTensor, vInput);
    copyHostToDevice(cuSeqLensTensor, cuSeqLens);

    cudaStream_t stream = nullptr;

    rt::launchFmhaReferenceCompact(
        qTensor, kTensor, vTensor, outputReference, cuSeqLensTensor, maxSeqLen, false, attentionScale, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    CuteDslFMHARunner runner(numHeads, numHeads, headDim);
    ASSERT_TRUE(runner.run(qTensor.dataPointer<half>(), kTensor.dataPointer<half>(), vTensor.dataPointer<half>(),
        outputCuteDsl.dataPointer<half>(), cuSeqLensTensor.dataPointer<int32_t>(), totalSeqLen, maxSeqLen, batchSize,
        stream, attentionScale));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    expectHalfOutputsClose(outputCuteDsl, outputReference,
        "ViT CuTe DSL FMHA headDim=" + std::to_string(headDim) + " numHeads=" + std::to_string(numHeads));
}

void runLlmAccuracyCase(int32_t batchSize, int32_t seqLen, int32_t numQHeads, int32_t numKVHeads, int32_t headDim,
    float attentionScale, float skipSoftmaxThresholdLog2 = 0.0F)
{
    bool const enableSkipSoftmax = skipSoftmaxThresholdLog2 < 0.0F;
    size_t const qSize = static_cast<size_t>(batchSize) * seqLen * numQHeads * headDim;
    size_t const kvSize = static_cast<size_t>(batchSize) * seqLen * numKVHeads * headDim;

    std::vector<half> qInput(qSize);
    std::vector<half> kInput(kvSize);
    std::vector<half> vInput(kvSize);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);

    rt::Tensor qCute({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor qReference({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kReference({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vReference({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice(qCute, qInput);
    copyHostToDevice(qReference, qInput);
    copyHostToDevice(kReference, kInput);
    copyHostToDevice(vReference, vInput);

    std::vector<half> kvContiguousInput(static_cast<size_t>(batchSize) * 2 * numKVHeads * seqLen * headDim);
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int32_t tokenIdx = 0; tokenIdx < seqLen; ++tokenIdx)
        {
            for (int32_t headIdx = 0; headIdx < numKVHeads; ++headIdx)
            {
                for (int32_t dimIdx = 0; dimIdx < headDim; ++dimIdx)
                {
                    size_t const srcOffset
                        = (((static_cast<size_t>(batchIdx) * seqLen + tokenIdx) * numKVHeads + headIdx) * headDim)
                        + dimIdx;
                    size_t const kOffset
                        = ((((static_cast<size_t>(batchIdx) * 2) * numKVHeads + headIdx) * seqLen + tokenIdx) * headDim)
                        + dimIdx;
                    size_t const vOffset
                        = (((((static_cast<size_t>(batchIdx) * 2 + 1) * numKVHeads + headIdx) * seqLen + tokenIdx)
                               * headDim)
                            + dimIdx);
                    kvContiguousInput[kOffset] = kInput[srcOffset];
                    kvContiguousInput[vOffset] = vInput[srcOffset];
                }
            }
        }
    }
    rt::Tensor kvCacheCute({batchSize, 2, numKVHeads, seqLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputReference({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputCuteDsl({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuKVSeqLens({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(kvCacheCute, kvContiguousInput);
    CUDA_CHECK(cudaMemset(outputReference.rawPointer(), 0, outputReference.getShape().volume() * sizeof(half)));
    CUDA_CHECK(cudaMemset(outputCuteDsl.rawPointer(), 0, outputCuteDsl.getShape().volume() * sizeof(half)));

    std::vector<int32_t> cuKVSeqLensHost(static_cast<size_t>(batchSize + 1));
    for (int32_t idx = 0; idx <= batchSize; ++idx)
    {
        cuKVSeqLensHost[static_cast<size_t>(idx)] = idx * seqLen;
    }

    copyHostToDevice(cuKVSeqLens, cuKVSeqLensHost);

    cudaStream_t stream = nullptr;

    CuteDslFMHARunner runner(numQHeads, numKVHeads, headDim, batchSize, seqLen, seqLen);
    ASSERT_TRUE(runner.run(qCute.dataPointer<half>(), kvCacheCute.dataPointer<half>(),
        outputCuteDsl.dataPointer<half>(), cuKVSeqLens.dataPointer<int32_t>(), stream, attentionScale, INT_MAX, false,
        1.0F, 1.0F, 1.0F, skipSoftmaxThresholdLog2));

    rt::launchFmhaReferenceBshd(qReference, kReference, vReference, outputReference, true, attentionScale, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    std::string const label = std::string(enableSkipSoftmax ? "Skip-softmax " : "") + "LLM CuTe DSL FMHA batch="
        + std::to_string(batchSize) + " seqLen=" + std::to_string(seqLen) + " numQHeads=" + std::to_string(numQHeads)
        + " numKVHeads=" + std::to_string(numKVHeads) + " headDim=" + std::to_string(headDim);
    if (enableSkipSoftmax)
    {
        expectSkipSoftmaxOutputsClose(outputCuteDsl, outputReference, label);
    }
    else
    {
        expectHalfOutputsClose(outputCuteDsl, outputReference, label);
    }
}

size_t contiguousKVIdx(int32_t b, int32_t kv, int32_t h, int32_t s, int32_t d, int32_t H, int32_t S, int32_t D)
{
    return (((((size_t) b * 2 + kv) * H + h) * S + s) * D + d);
}

size_t pagedKVIdx(int32_t page, int32_t h, int32_t tokenInPage, int32_t d, int32_t H, int32_t tokensPerPage, int32_t D)
{
    return static_cast<size_t>(((static_cast<int64_t>(page) * tokensPerPage + tokenInPage) * H + h) * D + d);
}

rt::Coords pagedKVPoolShape(int32_t numPages, int32_t numKVHeads, int32_t tokensPerPage, int32_t headDim)
{
    return rt::Coords{numPages, tokensPerPage, numKVHeads, headDim};
}

size_t bshdIdx(int32_t b, int32_t s, int32_t h, int32_t d, int32_t S, int32_t H, int32_t D)
{
    return static_cast<size_t>(((static_cast<int64_t>(b) * S + s) * H + h) * D + d);
}

void fillNonCausalVarlenReference(std::vector<half> const& qInput, std::vector<half> const& kInput,
    std::vector<half> const& vInput, std::vector<half>& output, std::vector<int32_t> const& qSeqLens,
    std::vector<int32_t> const& kvSeqLens, int32_t physicalSeqLenQ, int32_t physicalSeqLenKV, int32_t numQHeads,
    int32_t numKVHeads, int32_t headDim, float attentionScale)
{
    int32_t const batchSize = static_cast<int32_t>(qSeqLens.size());
    ASSERT_EQ(qSeqLens.size(), kvSeqLens.size());
    ASSERT_EQ(numQHeads % numKVHeads, 0);
    int32_t const qHeadsPerKVHead = numQHeads / numKVHeads;
    std::fill(output.begin(), output.end(), __float2half(0.0F));

    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        ASSERT_GE(qSeqLens[static_cast<size_t>(batchIdx)], 0);
        ASSERT_LE(qSeqLens[static_cast<size_t>(batchIdx)], physicalSeqLenQ);
        ASSERT_GT(kvSeqLens[static_cast<size_t>(batchIdx)], 0);
        ASSERT_LE(kvSeqLens[static_cast<size_t>(batchIdx)], physicalSeqLenKV);

        int32_t const qSeqLen = qSeqLens[static_cast<size_t>(batchIdx)];
        int32_t const kvSeqLen = kvSeqLens[static_cast<size_t>(batchIdx)];
        std::vector<float> scores(static_cast<size_t>(kvSeqLen));
        for (int32_t qToken = 0; qToken < qSeqLen; ++qToken)
        {
            for (int32_t qHead = 0; qHead < numQHeads; ++qHead)
            {
                int32_t const kvHead = qHead / qHeadsPerKVHead;
                float maxScore = -std::numeric_limits<float>::infinity();
                for (int32_t kvToken = 0; kvToken < kvSeqLen; ++kvToken)
                {
                    float dot = 0.0F;
                    for (int32_t dim = 0; dim < headDim; ++dim)
                    {
                        dot += __half2float(
                                   qInput[bshdIdx(batchIdx, qToken, qHead, dim, physicalSeqLenQ, numQHeads, headDim)])
                            * __half2float(
                                kInput[bshdIdx(batchIdx, kvToken, kvHead, dim, physicalSeqLenKV, numKVHeads, headDim)]);
                    }
                    scores[static_cast<size_t>(kvToken)] = dot * attentionScale;
                    maxScore = std::max(maxScore, scores[static_cast<size_t>(kvToken)]);
                }

                float sum = 0.0F;
                for (float& score : scores)
                {
                    score = std::exp(score - maxScore);
                    sum += score;
                }
                ASSERT_GT(sum, 0.0F);

                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    float value = 0.0F;
                    for (int32_t kvToken = 0; kvToken < kvSeqLen; ++kvToken)
                    {
                        float const probability = scores[static_cast<size_t>(kvToken)] / sum;
                        value += probability
                            * __half2float(
                                vInput[bshdIdx(batchIdx, kvToken, kvHead, dim, physicalSeqLenKV, numKVHeads, headDim)]);
                    }
                    output[bshdIdx(batchIdx, qToken, qHead, dim, physicalSeqLenQ, numQHeads, headDim)]
                        = __float2half(value);
                }
            }
        }
    }
}

std::pair<std::vector<int32_t>, std::vector<int32_t>> buildVisionBlockRanges(
    std::vector<int32_t> const& blockIds, std::vector<int32_t> const& validSeqLens, int32_t physicalSeqLen)
{
    std::vector<int32_t> blockBegin(blockIds.size(), -1);
    std::vector<int32_t> blockEnd(blockIds.size(), -1);
    for (int32_t batchIdx = 0; batchIdx < static_cast<int32_t>(validSeqLens.size()); ++batchIdx)
    {
        int32_t const validSeqLen = validSeqLens[static_cast<size_t>(batchIdx)];
        int32_t token = 0;
        while (token < validSeqLen)
        {
            size_t const offset = static_cast<size_t>(batchIdx) * physicalSeqLen;
            int32_t const blockId = blockIds[offset + token];
            int32_t end = token + 1;
            while (end < validSeqLen && blockIds[offset + end] == blockId)
            {
                ++end;
            }
            if (blockId >= 0)
            {
                std::fill(blockBegin.begin() + static_cast<int64_t>(offset + token),
                    blockBegin.begin() + static_cast<int64_t>(offset + end), token);
                std::fill(blockEnd.begin() + static_cast<int64_t>(offset + token),
                    blockEnd.begin() + static_cast<int64_t>(offset + end), end - 1);
            }
            token = end;
        }
    }
    return {blockBegin, blockEnd};
}

std::vector<half> computePagedBidirectionalReference(std::vector<half> const& q, std::vector<half> const& k,
    std::vector<half> const& v, std::vector<int32_t> const& blockBegin, std::vector<int32_t> const& blockEnd,
    std::vector<int32_t> const& validSeqLens, int32_t physicalSeqLen, int32_t numQHeads, int32_t numKVHeads,
    int32_t headDim, float attentionScale, int32_t slidingWindowSize)
{
    std::vector<half> output(q.size(), __float2half(0.0F));
    int32_t const groupSize = numQHeads / numKVHeads;
    for (int32_t batchIdx = 0; batchIdx < static_cast<int32_t>(validSeqLens.size()); ++batchIdx)
    {
        int32_t const validSeqLen = validSeqLens[static_cast<size_t>(batchIdx)];
        for (int32_t qHead = 0; qHead < numQHeads; ++qHead)
        {
            int32_t const kvHead = qHead / groupSize;
            std::vector<float> probabilities(static_cast<size_t>(validSeqLen));
            for (int32_t query = 0; query < validSeqLen; ++query)
            {
                std::fill(probabilities.begin(), probabilities.end(), -INFINITY);
                size_t const rowIdx = static_cast<size_t>(batchIdx) * physicalSeqLen + query;
                int32_t const visionBegin = blockBegin[rowIdx];
                int32_t const visionEnd = blockEnd[rowIdx];
                float maxLogit = -INFINITY;
                for (int32_t key = 0; key < validSeqLen; ++key)
                {
                    bool causal = key <= query;
                    if (slidingWindowSize < INT_MAX)
                    {
                        causal = causal && key >= query - slidingWindowSize;
                    }
                    bool const sameVisionBlock = visionBegin >= 0 && key >= visionBegin && key <= visionEnd;
                    if (!causal && !sameVisionBlock)
                    {
                        continue;
                    }

                    float dot = 0.0F;
                    for (int32_t dim = 0; dim < headDim; ++dim)
                    {
                        dot += __half2float(q[bshdIdx(batchIdx, query, qHead, dim, physicalSeqLen, numQHeads, headDim)])
                            * __half2float(k[bshdIdx(batchIdx, key, kvHead, dim, physicalSeqLen, numKVHeads, headDim)]);
                    }
                    probabilities[static_cast<size_t>(key)] = dot * attentionScale;
                    maxLogit = std::max(maxLogit, probabilities[static_cast<size_t>(key)]);
                }

                float denominator = 0.0F;
                for (float& probability : probabilities)
                {
                    probability = std::isfinite(probability) ? std::exp(probability - maxLogit) : 0.0F;
                    denominator += probability;
                }
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    float value = 0.0F;
                    for (int32_t key = 0; key < validSeqLen; ++key)
                    {
                        value += probabilities[static_cast<size_t>(key)]
                            * __half2float(v[bshdIdx(batchIdx, key, kvHead, dim, physicalSeqLen, numKVHeads, headDim)]);
                    }
                    output[bshdIdx(batchIdx, query, qHead, dim, physicalSeqLen, numQHeads, headDim)]
                        = __float2half(value / denominator);
                }
            }
        }
    }
    return output;
}

void runLlmPagedNonCausalAccuracyCase(int32_t physicalSeqLenQ, int32_t numQHeads, int32_t numKVHeads, int32_t headDim,
    std::vector<int32_t> const& qSeqLens, std::vector<int32_t> const& kvSeqLens, bool fp8Input)
{
    constexpr int32_t kPhysicalSeqLenKV = 128;
    constexpr int32_t kTokensPerPage = 128;
    constexpr float kQScale = 0.03125F;
    constexpr float kKScale = 0.0625F;
    constexpr float kVScale = 0.125F;

    int32_t const batchSize = static_cast<int32_t>(qSeqLens.size());
    ASSERT_EQ(qSeqLens.size(), kvSeqLens.size());
    ASSERT_GT(batchSize, 0);
    int32_t constexpr kMaxPagesPerSeq = 1;
    int32_t constexpr kCapacity = kMaxPagesPerSeq * kTokensPerPage;
    int32_t constexpr kPhysicalPagesPerGroup = kMaxPagesPerSeq + 1;
    int32_t const numPages = batchSize * 2 * kPhysicalPagesPerGroup;

    size_t const qSize = static_cast<size_t>(batchSize) * physicalSeqLenQ * numQHeads * headDim;
    size_t const kvSize = static_cast<size_t>(batchSize) * kPhysicalSeqLenKV * numKVHeads * headDim;
    size_t const pagedKVSize = static_cast<size_t>(numPages) * kTokensPerPage * numKVHeads * headDim;

    std::vector<half> qHalf(qSize);
    std::vector<half> kHalf(kvSize);
    std::vector<half> vHalf(kvSize);
    std::vector<__nv_fp8_e4m3> qFp8(qSize);
    std::vector<__nv_fp8_e4m3> kFp8(kvSize);
    std::vector<__nv_fp8_e4m3> vFp8(kvSize);
    std::vector<half> outputReferenceHost(qSize);
    std::mt19937 generator{20260723};
    std::uniform_real_distribution<float> distribution{-0.25F, 0.25F};

    auto fillValue = [&](float value, float scale, __nv_fp8_e4m3& fp8Value) {
        if (fp8Input)
        {
            fp8Value = __nv_fp8_e4m3{value / scale};
            return __float2half(static_cast<float>(fp8Value) * scale);
        }
        fp8Value = __nv_fp8_e4m3{0.0F};
        return __float2half(value);
    };

    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int32_t token = 0; token < physicalSeqLenQ; ++token)
        {
            for (int32_t head = 0; head < numQHeads; ++head)
            {
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    size_t const idx = bshdIdx(batchIdx, token, head, dim, physicalSeqLenQ, numQHeads, headDim);
                    float const value
                        = token < qSeqLens[static_cast<size_t>(batchIdx)] ? distribution(generator) : 7.0F;
                    qHalf[idx] = fillValue(value, kQScale, qFp8[idx]);
                }
            }
        }
        for (int32_t token = 0; token < kPhysicalSeqLenKV; ++token)
        {
            for (int32_t head = 0; head < numKVHeads; ++head)
            {
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    size_t const idx = bshdIdx(batchIdx, token, head, dim, kPhysicalSeqLenKV, numKVHeads, headDim);
                    bool const validKV = token < kvSeqLens[static_cast<size_t>(batchIdx)];
                    kHalf[idx] = fillValue(validKV ? distribution(generator) : -9.0F, kKScale, kFp8[idx]);
                    vHalf[idx] = fillValue(validKV ? distribution(generator) : 11.0F, kVScale, vFp8[idx]);
                }
            }
        }
    }

    std::vector<half> kvPagedHalf(pagedKVSize, __float2half(-13.0F));
    std::vector<__nv_fp8_e4m3> kvPagedFp8(pagedKVSize, __nv_fp8_e4m3{-13.0F});
    std::vector<int32_t> pageList(static_cast<size_t>(batchSize) * 2 * kMaxPagesPerSeq);
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int32_t kv = 0; kv < 2; ++kv)
        {
            int32_t const physicalBase = (batchIdx * 2 + kv) * kPhysicalPagesPerGroup;
            pageList[(batchIdx * 2 + kv) * kMaxPagesPerSeq] = physicalBase + 1;
        }
        for (int32_t kv = 0; kv < 2; ++kv)
        {
            for (int32_t token = 0; token < kvSeqLens[static_cast<size_t>(batchIdx)]; ++token)
            {
                int32_t const physicalPage = pageList[(batchIdx * 2 + kv) * kMaxPagesPerSeq];
                for (int32_t head = 0; head < numKVHeads; ++head)
                {
                    for (int32_t dim = 0; dim < headDim; ++dim)
                    {
                        size_t const sourceIdx
                            = bshdIdx(batchIdx, token, head, dim, kPhysicalSeqLenKV, numKVHeads, headDim);
                        size_t const pageIdx
                            = pagedKVIdx(physicalPage, head, token, dim, numKVHeads, kTokensPerPage, headDim);
                        kvPagedHalf[pageIdx] = kv == 0 ? kHalf[sourceIdx] : vHalf[sourceIdx];
                        kvPagedFp8[pageIdx] = kv == 0 ? kFp8[sourceIdx] : vFp8[sourceIdx];
                    }
                }
            }
        }
    }

    std::vector<int32_t> cuKVSeqLens(static_cast<size_t>(batchSize + 1));
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        cuKVSeqLens[static_cast<size_t>(batchIdx + 1)]
            = cuKVSeqLens[static_cast<size_t>(batchIdx)] + kvSeqLens[static_cast<size_t>(batchIdx)];
    }

    float const attentionScale = 1.0F / std::sqrt(static_cast<float>(headDim));
    fillNonCausalVarlenReference(qHalf, kHalf, vHalf, outputReferenceHost, qSeqLens, kvSeqLens, physicalSeqLenQ,
        kPhysicalSeqLenKV, numQHeads, numKVHeads, headDim, attentionScale);

    rt::Tensor qHalfTensor({batchSize, physicalSeqLenQ, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvHalfTensor(
        pagedKVPoolShape(numPages, numKVHeads, kTokensPerPage, headDim), rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor pageListTensor({batchSize, 2, kMaxPagesPerSeq}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor outputFp16({batchSize, physicalSeqLenQ, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputReference({batchSize, physicalSeqLenQ, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice(qHalfTensor, qHalf);
    copyHostToDevice(kvHalfTensor, kvPagedHalf);
    copyHostToDevice(pageListTensor, pageList);
    copyHostToDevice(cuKVSeqLensTensor, cuKVSeqLens);
    copyHostToDevice(outputReference, outputReferenceHost);

    cudaStream_t stream = nullptr;
    CuteDslFMHARunner runner(numQHeads, numKVHeads, headDim, batchSize, physicalSeqLenQ, kCapacity);
    ASSERT_TRUE(runner.runPaged(qHalfTensor.rawPointer(), kvHalfTensor.rawPointer(),
        pageListTensor.dataPointer<int32_t>(), outputFp16.rawPointer(), cuKVSeqLensTensor.dataPointer<int32_t>(),
        numPages, kMaxPagesPerSeq, kTokensPerPage, DataType::kHALF, stream, attentionScale, INT_MAX, /*fp8Input=*/false,
        1.0F, 1.0F, 1.0F, /*isCausal=*/false));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    std::string const labelPrefix = "D" + std::to_string(headDim) + " paged non-causal ";
    expectHalfOutputRowsClose(outputFp16, outputReference, qSeqLens, labelPrefix + "FP16 CuTe DSL FMHA");

    if (!fp8Input)
    {
        return;
    }

    rt::Tensor qFp8Tensor({batchSize, physicalSeqLenQ, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kFP8);
    rt::Tensor kvFp8Tensor(
        pagedKVPoolShape(numPages, numKVHeads, kTokensPerPage, headDim), rt::DeviceType::kGPU, DataType::kFP8);
    rt::Tensor outputFp8({batchSize, physicalSeqLenQ, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(qFp8Tensor, qFp8);
    copyHostToDevice(kvFp8Tensor, kvPagedFp8);

    ASSERT_TRUE(
        runner.runPaged(qFp8Tensor.rawPointer(), kvFp8Tensor.rawPointer(), pageListTensor.dataPointer<int32_t>(),
            outputFp8.rawPointer(), cuKVSeqLensTensor.dataPointer<int32_t>(), numPages, kMaxPagesPerSeq, kTokensPerPage,
            DataType::kFP8, stream, attentionScale, INT_MAX, /*fp8Input=*/true, kQScale, kKScale, kVScale,
            /*isCausal=*/false));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    expectFp8HalfOutputRowsClose(outputFp8, outputFp16, qSeqLens, labelPrefix + "FP8 CuTe DSL FMHA");
}

void runLlmD512PagedAccuracyCase(int32_t physicalSeqLen, int32_t numQHeads, int32_t numKVHeads,
    std::vector<int32_t> const& validSeqLens, float attentionScale, int32_t slidingWindowSize = INT_MAX,
    std::vector<int32_t> const* blockBeginHost = nullptr, std::vector<int32_t> const* blockEndHost = nullptr,
    bool validatePairedBlockPointers = false)
{
    constexpr int32_t kHeadDim = 512;
    constexpr int32_t kTokensPerPage = 128;
    int32_t const batchSize = static_cast<int32_t>(validSeqLens.size());
    int32_t const maxPagesPerSeq = (physicalSeqLen + kTokensPerPage - 1) / kTokensPerPage;
    int32_t const capacity = maxPagesPerSeq * kTokensPerPage;
    bool const useBidirectional = blockBeginHost != nullptr || blockEndHost != nullptr;
    ASSERT_EQ(blockBeginHost != nullptr, blockEndHost != nullptr);
    if (useBidirectional)
    {
        ASSERT_EQ(blockBeginHost->size(), static_cast<size_t>(batchSize) * physicalSeqLen);
        ASSERT_EQ(blockEndHost->size(), blockBeginHost->size());
    }

    // Give each batch/K-or-V group one spare physical page. Mapping logical
    // page i to physical page i+1 stays non-identity even for a one-page case,
    // while page zero in every group remains poisoned and unreachable.
    int32_t const physicalPagesPerGroup = maxPagesPerSeq + 1;
    int32_t const numPages = batchSize * 2 * physicalPagesPerGroup;

    size_t const qSize = static_cast<size_t>(batchSize) * physicalSeqLen * numQHeads * kHeadDim;
    size_t const kvSize = static_cast<size_t>(batchSize) * physicalSeqLen * numKVHeads * kHeadDim;
    size_t const pagedKVSize = static_cast<size_t>(numPages) * kTokensPerPage * numKVHeads * kHeadDim;

    std::vector<half> qInput(qSize);
    std::vector<half> kInput(kvSize);
    std::vector<half> vInput(kvSize);
    uniformFloatInitialization(qInput, -1.0F, 1.0F);
    uniformFloatInitialization(kInput, -1.0F, 1.0F);
    uniformFloatInitialization(vInput, -1.0F, 1.0F);

    half const qPoison = __float2half(113.0F);
    half const kPoison = __float2half(-127.0F);
    half const vPoison = __float2half(109.0F);
    std::vector<half> kvPaged(pagedKVSize, kPoison);
    std::vector<int32_t> pageList(static_cast<size_t>(batchSize) * 2 * maxPagesPerSeq);

    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        ASSERT_GT(validSeqLens[static_cast<size_t>(batchIdx)], 0);
        ASSERT_LE(validSeqLens[static_cast<size_t>(batchIdx)], physicalSeqLen);
        for (int32_t token = validSeqLens[static_cast<size_t>(batchIdx)]; token < physicalSeqLen; ++token)
        {
            for (int32_t head = 0; head < numQHeads; ++head)
            {
                for (int32_t dim = 0; dim < kHeadDim; ++dim)
                {
                    qInput[bshdIdx(batchIdx, token, head, dim, physicalSeqLen, numQHeads, kHeadDim)] = qPoison;
                }
            }
            for (int32_t head = 0; head < numKVHeads; ++head)
            {
                for (int32_t dim = 0; dim < kHeadDim; ++dim)
                {
                    size_t const idx = bshdIdx(batchIdx, token, head, dim, physicalSeqLen, numKVHeads, kHeadDim);
                    kInput[idx] = kPoison;
                    vInput[idx] = vPoison;
                }
            }
        }

        for (int32_t kv = 0; kv < 2; ++kv)
        {
            int32_t const physicalBase = (batchIdx * 2 + kv) * physicalPagesPerGroup;
            for (int32_t logicalPage = 0; logicalPage < maxPagesPerSeq; ++logicalPage)
            {
                pageList[(batchIdx * 2 + kv) * maxPagesPerSeq + logicalPage] = physicalBase + logicalPage + 1;
            }
        }
    }

    // Use a distinct poison value for the V half of every physical page. Any
    // wrong K/V page-table plane or unused-token read is therefore visible.
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const vBase = (batchIdx * 2 + 1) * physicalPagesPerGroup;
        for (int32_t page = 0; page < physicalPagesPerGroup; ++page)
        {
            for (int32_t token = 0; token < kTokensPerPage; ++token)
            {
                for (int32_t head = 0; head < numKVHeads; ++head)
                {
                    for (int32_t dim = 0; dim < kHeadDim; ++dim)
                    {
                        kvPaged[pagedKVIdx(vBase + page, head, token, dim, numKVHeads, kTokensPerPage, kHeadDim)]
                            = vPoison;
                    }
                }
            }
        }
    }

    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t const validSeqLen = validSeqLens[static_cast<size_t>(batchIdx)];
        for (int32_t kv = 0; kv < 2; ++kv)
        {
            std::vector<half> const& source = kv == 0 ? kInput : vInput;
            for (int32_t token = 0; token < validSeqLen; ++token)
            {
                int32_t const logicalPage = token / kTokensPerPage;
                int32_t const tokenInPage = token % kTokensPerPage;
                int32_t const physicalPage = pageList[(batchIdx * 2 + kv) * maxPagesPerSeq + logicalPage];
                for (int32_t head = 0; head < numKVHeads; ++head)
                {
                    for (int32_t dim = 0; dim < kHeadDim; ++dim)
                    {
                        kvPaged[pagedKVIdx(physicalPage, head, tokenInPage, dim, numKVHeads, kTokensPerPage, kHeadDim)]
                            = source[bshdIdx(batchIdx, token, head, dim, physicalSeqLen, numKVHeads, kHeadDim)];
                    }
                }
            }
        }
    }

    rt::Tensor qTensor({batchSize, physicalSeqLen, numQHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kTensor({batchSize, physicalSeqLen, numKVHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({batchSize, physicalSeqLen, numKVHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvPagedTensor(
        pagedKVPoolShape(numPages, numKVHeads, kTokensPerPage, kHeadDim), rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor pageListTensor({batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor outputReference({batchSize, physicalSeqLen, numQHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputCuteDsl({batchSize, physicalSeqLen, numQHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor paddedCuKVSeqLens({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor blockBeginTensor{};
    rt::Tensor blockEndTensor{};

    std::vector<int32_t> paddedCuKVSeqLensHost(static_cast<size_t>(batchSize + 1));
    for (int32_t batchIdx = 0; batchIdx <= batchSize; ++batchIdx)
    {
        // Q is batch-strided at physicalSeqLen. Match the plugin's padded
        // bottom-right contract; valid leading rows cannot attend poisoned
        // K/V padding because the attention remains causal.
        paddedCuKVSeqLensHost[static_cast<size_t>(batchIdx)] = batchIdx * physicalSeqLen;
    }

    copyHostToDevice(qTensor, qInput);
    copyHostToDevice(kTensor, kInput);
    copyHostToDevice(vTensor, vInput);
    copyHostToDevice(kvPagedTensor, kvPaged);
    copyHostToDevice(pageListTensor, pageList);
    copyHostToDevice(paddedCuKVSeqLens, paddedCuKVSeqLensHost);
    CUDA_CHECK(cudaMemset(outputReference.rawPointer(), 0, outputReference.getShape().volume() * sizeof(half)));
    CUDA_CHECK(cudaMemset(outputCuteDsl.rawPointer(), 0, outputCuteDsl.getShape().volume() * sizeof(half)));
    int32_t const* blockBegin = nullptr;
    int32_t const* blockEnd = nullptr;
    if (useBidirectional)
    {
        blockBeginTensor = rt::Tensor({batchSize, physicalSeqLen}, rt::DeviceType::kGPU, DataType::kINT32);
        blockEndTensor = rt::Tensor({batchSize, physicalSeqLen}, rt::DeviceType::kGPU, DataType::kINT32);
        copyHostToDevice(blockBeginTensor, *blockBeginHost);
        copyHostToDevice(blockEndTensor, *blockEndHost);
        blockBegin = blockBeginTensor.dataPointer<int32_t>();
        blockEnd = blockEndTensor.dataPointer<int32_t>();
    }

    cudaStream_t stream = nullptr;
    CuteDslFMHARunner runner(numQHeads, numKVHeads, kHeadDim, batchSize, physicalSeqLen, capacity);
    if (validatePairedBlockPointers)
    {
        EXPECT_THROW(
            runner.runPaged(qTensor.rawPointer(), kvPagedTensor.rawPointer(), pageListTensor.dataPointer<int32_t>(),
                outputCuteDsl.rawPointer(), paddedCuKVSeqLens.dataPointer<int32_t>(), numPages, maxPagesPerSeq,
                kTokensPerPage, DataType::kHALF, stream, attentionScale, slidingWindowSize, false, 1.0F, 1.0F, 1.0F,
                /*isCausal=*/true, /*skipSoftmaxThresholdLog2=*/0.0F, blockBegin, nullptr),
            std::runtime_error);
        EXPECT_THROW(
            runner.runPaged(qTensor.rawPointer(), kvPagedTensor.rawPointer(), pageListTensor.dataPointer<int32_t>(),
                outputCuteDsl.rawPointer(), paddedCuKVSeqLens.dataPointer<int32_t>(), numPages, maxPagesPerSeq,
                kTokensPerPage, DataType::kHALF, stream, attentionScale, slidingWindowSize, false, 1.0F, 1.0F, 1.0F,
                /*isCausal=*/true, /*skipSoftmaxThresholdLog2=*/0.0F, nullptr, blockEnd),
            std::runtime_error);
    }
    ASSERT_TRUE(runner.runPaged(qTensor.rawPointer(), kvPagedTensor.rawPointer(), pageListTensor.dataPointer<int32_t>(),
        outputCuteDsl.rawPointer(), paddedCuKVSeqLens.dataPointer<int32_t>(), numPages, maxPagesPerSeq, kTokensPerPage,
        DataType::kHALF, stream, attentionScale, slidingWindowSize, false, 1.0F, 1.0F, 1.0F,
        /*isCausal=*/true, /*skipSoftmaxThresholdLog2=*/0.0F, blockBegin, blockEnd));
    if (useBidirectional)
    {
        auto const reference
            = computePagedBidirectionalReference(qInput, kInput, vInput, *blockBeginHost, *blockEndHost, validSeqLens,
                physicalSeqLen, numQHeads, numKVHeads, kHeadDim, attentionScale, slidingWindowSize);
        copyHostToDevice(outputReference, reference);
    }
    else
    {
        rt::launchFmhaReferenceBshd(qTensor, kTensor, vTensor, outputReference, true, attentionScale, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    expectHalfOutputRowsClose(outputCuteDsl, outputReference, validSeqLens,
        "D512 paged LLM CuTe DSL FMHA physicalSeqLen=" + std::to_string(physicalSeqLen) + " numQHeads="
            + std::to_string(numQHeads) + " numKVHeads=" + std::to_string(numKVHeads) + " bidirectional="
            + std::to_string(useBidirectional) + " slidingWindowSize=" + std::to_string(slidingWindowSize));
}

void runLlmD512PagedFp8AccuracyCase(int32_t seqLen, int32_t slidingWindowSize)
{
    constexpr int32_t kBatchSize = 1;
    constexpr int32_t kNumQHeads = 4;
    constexpr int32_t kNumKVHeads = 2;
    constexpr int32_t kHeadDim = 512;
    constexpr int32_t kTokensPerPage = 128;
    constexpr float kQScale = 0.03125F;
    constexpr float kKScale = 0.0625F;
    constexpr float kVScale = 0.125F;
    ASSERT_EQ(seqLen % kTokensPerPage, 0);

    int32_t const maxPagesPerSeq = seqLen / kTokensPerPage;
    int32_t const physicalPagesPerGroup = maxPagesPerSeq + 1;
    int32_t const numPages = 2 * physicalPagesPerGroup;
    size_t const qSize = static_cast<size_t>(seqLen) * kNumQHeads * kHeadDim;
    size_t const pagedKVSize = static_cast<size_t>(numPages) * kTokensPerPage * kNumKVHeads * kHeadDim;

    std::vector<__nv_fp8_e4m3> qFp8(qSize);
    std::vector<half> qFp16(qSize);
    std::vector<__nv_fp8_e4m3> kvFp8(pagedKVSize, __nv_fp8_e4m3{17.0F});
    std::vector<half> kvFp16(pagedKVSize, __float2half(17.0F));
    std::vector<int32_t> pageList(static_cast<size_t>(2 * maxPagesPerSeq));
    std::mt19937 generator{20260722};
    std::uniform_real_distribution<float> distribution{-1.0F, 1.0F};

    for (size_t idx = 0; idx < qSize; ++idx)
    {
        __nv_fp8_e4m3 const quantized{distribution(generator) / kQScale};
        qFp8[idx] = quantized;
        qFp16[idx] = __float2half(static_cast<float>(quantized) * kQScale);
    }
    for (int32_t kv = 0; kv < 2; ++kv)
    {
        float const scale = kv == 0 ? kKScale : kVScale;
        int32_t const physicalBase = kv * physicalPagesPerGroup;
        for (int32_t logicalPage = 0; logicalPage < maxPagesPerSeq; ++logicalPage)
        {
            int32_t const physicalPage = physicalBase + logicalPage + 1;
            pageList[kv * maxPagesPerSeq + logicalPage] = physicalPage;
            for (int32_t token = 0; token < kTokensPerPage; ++token)
            {
                for (int32_t head = 0; head < kNumKVHeads; ++head)
                {
                    for (int32_t dim = 0; dim < kHeadDim; ++dim)
                    {
                        size_t const idx
                            = pagedKVIdx(physicalPage, head, token, dim, kNumKVHeads, kTokensPerPage, kHeadDim);
                        __nv_fp8_e4m3 const quantized{distribution(generator) / scale};
                        kvFp8[idx] = quantized;
                        kvFp16[idx] = __float2half(static_cast<float>(quantized) * scale);
                    }
                }
            }
        }
    }

    rt::Tensor qFp8Tensor({kBatchSize, seqLen, kNumQHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kFP8);
    rt::Tensor qFp16Tensor({kBatchSize, seqLen, kNumQHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvFp8Tensor(
        pagedKVPoolShape(numPages, kNumKVHeads, kTokensPerPage, kHeadDim), rt::DeviceType::kGPU, DataType::kFP8);
    rt::Tensor kvFp16Tensor(
        pagedKVPoolShape(numPages, kNumKVHeads, kTokensPerPage, kHeadDim), rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor pageListTensor({kBatchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLens({2}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor outputFp8({kBatchSize, seqLen, kNumQHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputFp16({kBatchSize, seqLen, kNumQHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice(qFp8Tensor, qFp8);
    copyHostToDevice(qFp16Tensor, qFp16);
    copyHostToDevice(kvFp8Tensor, kvFp8);
    copyHostToDevice(kvFp16Tensor, kvFp16);
    copyHostToDevice(pageListTensor, pageList);
    copyHostToDevice(cuKVSeqLens, std::vector<int32_t>{0, seqLen});

    cudaStream_t stream = nullptr;
    float const attentionScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    CuteDslFMHARunner runner(kNumQHeads, kNumKVHeads, kHeadDim, kBatchSize, seqLen, seqLen);
    ASSERT_TRUE(runner.runPaged(qFp16Tensor.rawPointer(), kvFp16Tensor.rawPointer(),
        pageListTensor.dataPointer<int32_t>(), outputFp16.rawPointer(), cuKVSeqLens.dataPointer<int32_t>(), numPages,
        maxPagesPerSeq, kTokensPerPage, DataType::kHALF, stream, attentionScale, slidingWindowSize));
    ASSERT_TRUE(
        runner.runPaged(qFp8Tensor.rawPointer(), kvFp8Tensor.rawPointer(), pageListTensor.dataPointer<int32_t>(),
            outputFp8.rawPointer(), cuKVSeqLens.dataPointer<int32_t>(), numPages, maxPagesPerSeq, kTokensPerPage,
            DataType::kFP8, stream, attentionScale, slidingWindowSize, true, kQScale, kKScale, kVScale));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    expectFp8HalfOutputsClose(outputFp8, outputFp16,
        "D512 paged FP8 LLM CuTe DSL FMHA seqLen=" + std::to_string(seqLen)
            + " slidingWindowSize=" + std::to_string(slidingWindowSize));
}

template <typename T>
void runLlmPagedMatchesContiguousCase(int32_t batchSize, int32_t seqLen, int32_t numQHeads, int32_t numKVHeads,
    int32_t headDim, int32_t tokensPerPage, DataType dataType, int32_t slidingWindowSize = INT_MAX,
    bool fp8Input = false, float qScale = 1.0F, float kScale = 1.0F, float vScale = 1.0F,
    float skipSoftmaxThresholdLog2 = 0.0F)
{
    ASSERT_EQ(seqLen % tokensPerPage, 0);
    int32_t const maxPagesPerSeq = seqLen / tokensPerPage;
    int32_t const numPages = batchSize * 2 * maxPagesPerSeq;

    size_t const qSize = static_cast<size_t>(batchSize) * seqLen * numQHeads * headDim;
    size_t const kvSize = static_cast<size_t>(batchSize) * 2 * numKVHeads * seqLen * headDim;
    size_t const pagedKVSize = static_cast<size_t>(numPages) * numKVHeads * tokensPerPage * headDim;

    std::vector<T> qInput(qSize);
    std::vector<T> kvContiguous(kvSize);
    std::vector<T> kvPaged(pagedKVSize);
    std::vector<int32_t> pageList(static_cast<size_t>(batchSize) * 2 * maxPagesPerSeq);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kvContiguous, -1.0f, 1.0f);

    for (int32_t b = 0; b < batchSize; ++b)
    {
        int32_t const batchPageBase = b * 2 * maxPagesPerSeq;
        for (int32_t logicalPage = 0; logicalPage < maxPagesPerSeq; ++logicalPage)
        {
            int32_t const permutedPage = (logicalPage * 3 + 1) % maxPagesPerSeq;
            pageList[(b * 2 * maxPagesPerSeq) + logicalPage] = batchPageBase + permutedPage;
            pageList[(b * 2 * maxPagesPerSeq) + maxPagesPerSeq + logicalPage]
                = batchPageBase + maxPagesPerSeq + permutedPage;
        }
    }

    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t kv = 0; kv < 2; ++kv)
        {
            for (int32_t s = 0; s < seqLen; ++s)
            {
                int32_t const logicalPage = s / tokensPerPage;
                int32_t const tokenInPage = s % tokensPerPage;
                int32_t const page = pageList[(b * 2 + kv) * maxPagesPerSeq + logicalPage];
                for (int32_t h = 0; h < numKVHeads; ++h)
                {
                    for (int32_t d = 0; d < headDim; ++d)
                    {
                        kvPaged[pagedKVIdx(page, h, tokenInPage, d, numKVHeads, tokensPerPage, headDim)]
                            = kvContiguous[contiguousKVIdx(b, kv, h, s, d, numKVHeads, seqLen, headDim)];
                    }
                }
            }
        }
    }

    rt::Tensor qContiguous({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, dataType);
    rt::Tensor qPaged({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, dataType);
    rt::Tensor kvContiguousTensor({batchSize, 2, numKVHeads, seqLen, headDim}, rt::DeviceType::kGPU, dataType);
    rt::Tensor kvPagedTensor(
        pagedKVPoolShape(numPages, numKVHeads, tokensPerPage, headDim), rt::DeviceType::kGPU, dataType);
    rt::Tensor pageListTensor({batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor outputContiguous({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputPaged({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuKVSeqLens({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    std::vector<int32_t> cuKVSeqLensHost(static_cast<size_t>(batchSize + 1));
    for (int32_t idx = 0; idx <= batchSize; ++idx)
    {
        cuKVSeqLensHost[static_cast<size_t>(idx)] = idx * seqLen;
    }

    copyHostToDevice(qContiguous, qInput);
    copyHostToDevice(qPaged, qInput);
    copyHostToDevice(kvContiguousTensor, kvContiguous);
    copyHostToDevice(kvPagedTensor, kvPaged);
    copyHostToDevice(pageListTensor, pageList);
    copyHostToDevice(cuKVSeqLens, cuKVSeqLensHost);

    cudaStream_t stream = nullptr;
    float const attentionScale = 1.0F / std::sqrt(static_cast<float>(headDim));
    CuteDslFMHARunner runner(numQHeads, numKVHeads, headDim, batchSize, seqLen, seqLen);
    ASSERT_TRUE(runner.run(qContiguous.rawPointer(), kvContiguousTensor.rawPointer(),
        outputContiguous.dataPointer<half>(), cuKVSeqLens.dataPointer<int32_t>(), stream, attentionScale,
        slidingWindowSize, fp8Input, qScale, kScale, vScale, skipSoftmaxThresholdLog2));
    ASSERT_TRUE(runner.runPaged(qPaged.rawPointer(), kvPagedTensor.rawPointer(), pageListTensor.dataPointer<int32_t>(),
        outputPaged.dataPointer<half>(), cuKVSeqLens.dataPointer<int32_t>(), numPages, maxPagesPerSeq, tokensPerPage,
        dataType, stream, attentionScale, slidingWindowSize, fp8Input, qScale, kScale, vScale, /*isCausal=*/true,
        skipSoftmaxThresholdLog2));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    expectHalfOutputsClose(outputPaged, outputContiguous,
        "Paged LLM CuTe DSL FMHA batch=" + std::to_string(batchSize) + " seqLen=" + std::to_string(seqLen)
            + " numQHeads=" + std::to_string(numQHeads) + " numKVHeads=" + std::to_string(numKVHeads)
            + " headDim=" + std::to_string(headDim) + " tokensPerPage=" + std::to_string(tokensPerPage)
            + " fp8Input=" + std::to_string(fp8Input) + " slidingWindowSize=" + std::to_string(slidingWindowSize));
}

void runLlmFp8LongSequenceAccuracyCase(int32_t numQHeads, int32_t numKVHeads, int32_t headDim)
{
    // Nemotron layer-0 scales. A 2048-token sequence exercises
    // multiple online-softmax tiles where FP8 skip-correction must remain representable.
    constexpr int32_t batchSize = 1;
    constexpr int32_t seqLen = 2048;
    constexpr float qScale = 0.01429094560444355f;
    constexpr float kScale = 1.0f;
    constexpr float vScale = 1.0f;

    size_t const qSize = static_cast<size_t>(batchSize) * seqLen * numQHeads * headDim;
    size_t const kvSize = static_cast<size_t>(batchSize) * seqLen * numKVHeads * headDim;
    size_t const kvCacheSize = 2 * kvSize;

    std::vector<__nv_fp8_e4m3> qFp8Host(qSize);
    std::vector<__nv_fp8_e4m3> kvCacheFp8Host(kvCacheSize);
    std::vector<half> qReferenceHost(qSize);
    std::vector<half> kReferenceHost(kvSize);
    std::vector<half> vReferenceHost(kvSize);

    // Dequantize the generated E4M3 values for the reference path so both
    // implementations receive identical quantized inputs.
    std::mt19937 generator{20260703};
    std::uniform_real_distribution<float> distribution{-6.0f, 6.0f};
    for (size_t index = 0; index < qSize; ++index)
    {
        __nv_fp8_e4m3 const quantized{distribution(generator) / qScale};
        qFp8Host[index] = quantized;
        qReferenceHost[index] = __float2half(static_cast<float>(quantized) * qScale);
    }
    for (int32_t token = 0; token < seqLen; ++token)
    {
        for (int32_t kvHead = 0; kvHead < numKVHeads; ++kvHead)
        {
            for (int32_t dim = 0; dim < headDim; ++dim)
            {
                __nv_fp8_e4m3 const quantizedK{distribution(generator) / kScale};
                __nv_fp8_e4m3 const quantizedV{distribution(generator) / vScale};
                size_t const referenceIndex = static_cast<size_t>((token * numKVHeads + kvHead) * headDim + dim);
                size_t const kCacheIndex = static_cast<size_t>((kvHead * seqLen + token) * headDim + dim);
                size_t const vCacheIndex
                    = static_cast<size_t>(((numKVHeads + kvHead) * seqLen + token) * headDim + dim);
                kvCacheFp8Host[kCacheIndex] = quantizedK;
                kvCacheFp8Host[vCacheIndex] = quantizedV;
                kReferenceHost[referenceIndex] = __float2half(static_cast<float>(quantizedK) * kScale);
                vReferenceHost[referenceIndex] = __float2half(static_cast<float>(quantizedV) * vScale);
            }
        }
    }

    rt::Tensor qFp8({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kFP8);
    rt::Tensor kvCacheFp8({batchSize, 2, numKVHeads, seqLen, headDim}, rt::DeviceType::kGPU, DataType::kFP8);
    rt::Tensor qReference({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kReference({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vReference({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputCuteDsl({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputReference({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuKVSeqLens({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(qFp8, qFp8Host);
    copyHostToDevice(kvCacheFp8, kvCacheFp8Host);
    copyHostToDevice(qReference, qReferenceHost);
    copyHostToDevice(kReference, kReferenceHost);
    copyHostToDevice(vReference, vReferenceHost);
    copyHostToDevice(cuKVSeqLens, std::vector<int32_t>{0, seqLen});

    cudaStream_t stream = nullptr;
    float const attentionScale = 1.0F / std::sqrt(static_cast<float>(headDim));
    CuteDslFMHARunner runner(numQHeads, numKVHeads, headDim, batchSize, seqLen, seqLen);
    ASSERT_TRUE(runner.run(qFp8.rawPointer(), kvCacheFp8.rawPointer(), outputCuteDsl.rawPointer(),
        cuKVSeqLens.dataPointer<int32_t>(), stream, attentionScale, INT_MAX, true, qScale, kScale, vScale));
    rt::launchFmhaReferenceBshd(qReference, kReference, vReference, outputReference, true, attentionScale, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    auto const actual = copyDeviceToHost<half>(outputCuteDsl);
    auto const expected = copyDeviceToHost<half>(outputReference);
    double sumAbsError = 0.0;
    double sumSquaredActual = 0.0;
    double sumSquaredExpected = 0.0;
    double dot = 0.0;
    bool nanDetected = false;
    for (size_t index = 0; index < qSize; ++index)
    {
        float const actualValue = __half2float(actual[index]);
        float const expectedValue = __half2float(expected[index]);
        sumAbsError += std::fabs(actualValue - expectedValue);
        sumSquaredActual += static_cast<double>(actualValue) * actualValue;
        sumSquaredExpected += static_cast<double>(expectedValue) * expectedValue;
        dot += static_cast<double>(actualValue) * expectedValue;
        nanDetected = nanDetected || std::isnan(actualValue);
    }

    double const meanAbsError = sumAbsError / qSize;
    double const cosineSimilarity = dot / std::sqrt(std::max(sumSquaredActual * sumSquaredExpected, 1.0e-30));
    EXPECT_FALSE(nanDetected);
    EXPECT_LT(meanAbsError, 0.05);
    EXPECT_GT(cosineSimilarity, 0.99);
}

} // namespace

TEST(CuteDslFMHARunnerTest, canImplement)
{
    EXPECT_TRUE(CuteDslFMHARunner::canImplement(64, 100));
    EXPECT_TRUE(CuteDslFMHARunner::canImplement(128, 101));
    EXPECT_TRUE(CuteDslFMHARunner::canImplement(256, 110));
    EXPECT_TRUE(CuteDslFMHARunner::canImplement(512, 100));
    EXPECT_TRUE(CuteDslFMHARunner::canImplement(512, 101));
    EXPECT_TRUE(CuteDslFMHARunner::canImplement(512, 110));
    EXPECT_FALSE(CuteDslFMHARunner::canImplement(256, 90));
    EXPECT_FALSE(CuteDslFMHARunner::canImplement(64, 103));
    EXPECT_FALSE(CuteDslFMHARunner::canImplement(128, 120));
    EXPECT_FALSE(CuteDslFMHARunner::canImplement(512, 90));
    EXPECT_FALSE(CuteDslFMHARunner::canImplement(512, 103));
    EXPECT_FALSE(CuteDslFMHARunner::canImplement(512, 120));
    EXPECT_FALSE(CuteDslFMHARunner::canImplement(512, 121));

    EXPECT_TRUE(CuteDslFMHARunner::canImplementViT(64, 100));
    EXPECT_TRUE(CuteDslFMHARunner::canImplementViT(128, 110));
    EXPECT_FALSE(CuteDslFMHARunner::canImplementViT(64, 103));
    EXPECT_FALSE(CuteDslFMHARunner::canImplementViT(128, 120));
}

TEST(CuteDslFMHARunnerTest, llmD512PagedAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "D512 CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    struct LlmD512Case
    {
        int32_t seqLen;
        int32_t numQHeads;
        int32_t numKVHeads;
        std::optional<float> attentionScale;
    };

    // Cover the 128-token page/query boundaries and Gemma4 E2B, E4B, and
    // 31B global-attention GQA geometries. Every case uses rounded capacity,
    // non-identity K/V page tables, and poisoned unused physical storage.
    std::vector<LlmD512Case> const cases{
        {127, 8, 1},
        {128, 8, 2, 0.37F},
        {129, 32, 4},
        {257, 8, 1},
    };

    for (auto const& testCase : cases)
    {
        SCOPED_TRACE(::testing::Message() << "seqLen=" << testCase.seqLen << " numQHeads=" << testCase.numQHeads
                                          << " numKVHeads=" << testCase.numKVHeads);
        float const attentionScale = testCase.attentionScale.value_or(1.0F / std::sqrt(static_cast<float>(512)));
        runLlmD512PagedAccuracyCase(
            testCase.seqLen, testCase.numQHeads, testCase.numKVHeads, {testCase.seqLen}, attentionScale);
    }
}

TEST(CuteDslFMHARunnerTest, llmD512PagedRaggedAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "D512 CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    std::vector<int32_t> const validSeqLens{1, 128, 257};
    runLlmD512PagedAccuracyCase(257, 8, 2, validSeqLens, 1.0F / std::sqrt(static_cast<float>(512)));
}

TEST(CuteDslFMHARunnerTest, llmD512PagedNonCausalAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "D512 CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }
    runLlmPagedNonCausalAccuracyCase(32, 8, 1, 512, /*qSeqLens=*/{17, 9}, /*kvSeqLens=*/{96, 37}, /*fp8Input=*/false);
}

TEST(CuteDslFMHARunnerTest, llmD256PagedNonCausalAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "D256 CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    runLlmPagedNonCausalAccuracyCase(4, 16, 8, 256, /*qSeqLens=*/{4, 3}, /*kvSeqLens=*/{24, 11}, /*fp8Input=*/false);
}

TEST(CuteDslFMHARunnerTest, llmD512PagedBidirectionalAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "D512 CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }
    constexpr int32_t kNumQHeads = 4;
    constexpr int32_t kNumKVHeads = 2;
    float const attentionScale = 1.0F / std::sqrt(512.0F);

    // All-text sentinels must be identical to ordinary full-causal attention.
    {
        constexpr int32_t kSeqLen = 65;
        std::vector<int32_t> const validSeqLens{kSeqLen};
        std::vector<int32_t> const blockIds(kSeqLen, -1);
        auto const [blockBegin, blockEnd] = buildVisionBlockRanges(blockIds, validSeqLens, kSeqLen);
        runLlmD512PagedAccuracyCase(
            kSeqLen, kNumQHeads, kNumKVHeads, validSeqLens, attentionScale, INT_MAX, &blockBegin, &blockEnd);
    }

    // The same sliding-capable artifact must preserve an active future-facing
    // block when the global layer passes the no-limit sentinel.
    {
        constexpr int32_t kSeqLen = 130;
        std::vector<int32_t> const validSeqLens{kSeqLen};
        std::vector<int32_t> blockIds(kSeqLen, -1);
        for (int32_t token = 120; token < kSeqLen; ++token)
        {
            blockIds[static_cast<size_t>(token)] = 0;
        }
        auto const [blockBegin, blockEnd] = buildVisionBlockRanges(blockIds, validSeqLens, kSeqLen);
        runLlmD512PagedAccuracyCase(
            kSeqLen, kNumQHeads, kNumKVHeads, validSeqLens, attentionScale, INT_MAX, &blockBegin, &blockEnd);
    }

    // Combine ragged batching, two disjoint blocks, a block crossing the
    // 128-token page boundary, sliding-causal masking, non-identity page
    // tables, and the paired-pointer runtime guard in one compact oracle case.
    {
        constexpr int32_t kSeqLen = 130;
        constexpr int32_t kWindowSizeLeft = 31; // 32 keys including the query.
        std::vector<int32_t> const validSeqLens{kSeqLen, 99};
        std::vector<int32_t> blockIds(static_cast<size_t>(validSeqLens.size()) * kSeqLen, -1);
        size_t const batchOneOffset = kSeqLen;
        for (int32_t token = 8; token < 32; ++token)
        {
            blockIds[static_cast<size_t>(token)] = 0;
        }
        for (int32_t token = 120; token < 130; ++token)
        {
            blockIds[static_cast<size_t>(token)] = 1;
        }
        for (int32_t token = 16; token < 48; ++token)
        {
            blockIds[batchOneOffset + static_cast<size_t>(token)] = 0;
        }
        for (int32_t token = 80; token < 99; ++token)
        {
            blockIds[batchOneOffset + static_cast<size_t>(token)] = 1;
        }
        auto const [blockBegin, blockEnd] = buildVisionBlockRanges(blockIds, validSeqLens, kSeqLen);
        runLlmD512PagedAccuracyCase(kSeqLen, kNumQHeads, kNumKVHeads, validSeqLens, attentionScale, kWindowSizeLeft,
            &blockBegin, &blockEnd, /*validatePairedBlockPointers=*/true);
    }
}

TEST(CuteDslFMHARunnerTest, llmD512PagedFp8Accuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "D512 CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }
    runLlmD512PagedFp8AccuracyCase(128, INT_MAX);
    runLlmD512PagedFp8AccuracyCase(128, 63);
    runLlmD512PagedFp8AccuracyCase(1024, INT_MAX);
}

TEST(CuteDslFMHARunnerTest, llmD512PagedFp8NonCausalAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "D512 CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }
    runLlmPagedNonCausalAccuracyCase(32, 8, 1, 512, /*qSeqLens=*/{17, 9}, /*kvSeqLens=*/{96, 37}, /*fp8Input=*/true);
}

TEST(CuteDslFMHARunnerTest, llmD256PagedFp8NonCausalAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "D256 CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }
    runLlmPagedNonCausalAccuracyCase(4, 16, 8, 256, /*qSeqLens=*/{4, 3}, /*kvSeqLens=*/{24, 11}, /*fp8Input=*/true);
}

TEST(CuteDslFMHARunnerTest, vitAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    struct ViTCase
    {
        std::vector<int32_t> cuSeqLens;
        int32_t numHeads;
        int32_t headDim;
        int32_t maxSeqLen;
        std::optional<float> attentionScale;
    };

    std::vector<ViTCase> const cases{
        {{0, 32, 60, 88, 128}, 14, 64, 128},
        {{0, 16, 64}, 14, 72, 128},
        {{0, 24, 80, 144}, 14, 80, 160},
        {{0, 40, 96, 200}, 14, 96, 256},
        {{0, 100, 200, 300}, 14, 128, 512},
        {{0, 16, 48}, 8, 64, 48, 1.0F},
        {{0, 24, 64}, 8, 80, 64, 0.37F},
    };

    for (auto const& testCase : cases)
    {
        std::string cuSeqLensStr = "[";
        for (size_t i = 0; i < testCase.cuSeqLens.size(); ++i)
        {
            if (i)
                cuSeqLensStr += ",";
            cuSeqLensStr += std::to_string(testCase.cuSeqLens[i]);
        }
        cuSeqLensStr += "]";
        SCOPED_TRACE(::testing::Message() << "numHeads=" << testCase.numHeads << " headDim=" << testCase.headDim
                                          << " maxSeqLen=" << testCase.maxSeqLen << " cuSeqLens=" << cuSeqLensStr);
        float const attentionScale
            = testCase.attentionScale.value_or(1.0F / std::sqrt(static_cast<float>(testCase.headDim)));
        runViTAccuracyCase(testCase.cuSeqLens, testCase.numHeads, testCase.headDim, testCase.maxSeqLen, attentionScale);
    }
}

TEST(CuteDslFMHARunnerTest, llmAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    struct LlmCase
    {
        int32_t batchSize;
        int32_t seqLen;
        int32_t numQHeads;
        int32_t numKVHeads;
        int32_t headDim;
        std::optional<float> attentionScale;
    };

    std::vector<LlmCase> const cases{
        {2, 32, 8, 8, 64},
        {2, 48, 16, 4, 64},
        {1, 24, 8, 8, 128},
        {1, 32, 12, 4, 128},
        {1, 16, 8, 8, 64, 1.0F},
        {1, 16, 8, 2, 128, 0.37F},
        {1, 32, 16, 2, 256},
        {1, 256, 16, 2, 256},
    };

    for (auto const& testCase : cases)
    {
        SCOPED_TRACE(::testing::Message() << "batchSize=" << testCase.batchSize << " seqLen=" << testCase.seqLen
                                          << " numQHeads=" << testCase.numQHeads
                                          << " numKVHeads=" << testCase.numKVHeads << " headDim=" << testCase.headDim);
        float const attentionScale
            = testCase.attentionScale.value_or(1.0F / std::sqrt(static_cast<float>(testCase.headDim)));
        runLlmAccuracyCase(testCase.batchSize, testCase.seqLen, testCase.numQHeads, testCase.numKVHeads,
            testCase.headDim, attentionScale);
    }
}

TEST(CuteDslFMHARunnerTest, llmSkipSoftmaxAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    struct LlmCase
    {
        int32_t batchSize;
        int32_t seqLen;
        int32_t numQHeads;
        int32_t numKVHeads;
        int32_t headDim;
    };

    // seqLen must span several 128-token KV tiles: single-tile rows can never
    // skip (first-tile rule), so a short sequence would not exercise the skip
    // predicate / vote / P*V-skip path at all.
    std::vector<LlmCase> const cases{
        {2, 1024, 14, 2, 64},
        {1, 1024, 16, 8, 128},
    };

    for (auto const& testCase : cases)
    {
        SCOPED_TRACE(::testing::Message() << "batchSize=" << testCase.batchSize << " seqLen=" << testCase.seqLen
                                          << " numQHeads=" << testCase.numQHeads
                                          << " numKVHeads=" << testCase.numKVHeads << " headDim=" << testCase.headDim);
        float const attentionScale = 1.0F / std::sqrt(static_cast<float>(testCase.headDim));
        // A defaulted 0.0 threshold trips the runner's validity guard and
        // silently dispatches dense; pass a real lambda.
        float const lambda = testCase.headDim == 64 ? 0.003F : 0.001F;
        runLlmAccuracyCase(testCase.batchSize, testCase.seqLen, testCase.numQHeads, testCase.numKVHeads,
            testCase.headDim, attentionScale, std::log2(lambda));
    }
}

TEST(CuteDslFMHARunnerTest, llmPagedKVMatchesContiguous)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    struct LlmPagedCase
    {
        int32_t batchSize;
        int32_t seqLen;
        int32_t numQHeads;
        int32_t numKVHeads;
        int32_t headDim;
        int32_t tokensPerPage;
        int32_t slidingWindowSize{INT_MAX};
    };

    std::vector<LlmPagedCase> const cases{
        {1, 256, 8, 8, 64, 128},
        {2, 128, 16, 4, 64, 128},
        {1, 128, 8, 8, 128, 128},
        {1, 256, 12, 4, 128, 128},
        {1, 256, 16, 2, 256, 128},
        {1, 256, 16, 2, 256, 128, 192},
    };

    for (auto const& testCase : cases)
    {
        SCOPED_TRACE(::testing::Message()
            << "batchSize=" << testCase.batchSize << " seqLen=" << testCase.seqLen
            << " numQHeads=" << testCase.numQHeads << " numKVHeads=" << testCase.numKVHeads
            << " headDim=" << testCase.headDim << " tokensPerPage=" << testCase.tokensPerPage);
        runLlmPagedMatchesContiguousCase<half>(testCase.batchSize, testCase.seqLen, testCase.numQHeads,
            testCase.numKVHeads, testCase.headDim, testCase.tokensPerPage, DataType::kHALF, testCase.slidingWindowSize);
    }
}

// Paged skip-softmax validation by transitivity: contiguous skip is already
// checked against a numpy reference (llmSkipSoftmaxAccuracy), so if the paged
// skip kernel skips the same tiles and produces the same output as the
// contiguous skip kernel, the paged skip integration is correct. Causal,
// FP16, non-sliding, d64/d128 only; seqLen spans several 128-token tiles so
// the skip predicate/vote/P*V-skip path is actually exercised.
TEST(CuteDslFMHARunnerTest, llmPagedKVSkipSoftmaxMatchesContiguous)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    struct LlmPagedSkipCase
    {
        int32_t batchSize;
        int32_t seqLen;
        int32_t numQHeads;
        int32_t numKVHeads;
        int32_t headDim;
    };

    std::vector<LlmPagedSkipCase> const cases{
        {2, 1024, 14, 2, 64},
        {1, 1024, 16, 8, 128},
    };

    for (auto const& testCase : cases)
    {
        SCOPED_TRACE(::testing::Message() << "batchSize=" << testCase.batchSize << " seqLen=" << testCase.seqLen
                                          << " numQHeads=" << testCase.numQHeads
                                          << " numKVHeads=" << testCase.numKVHeads << " headDim=" << testCase.headDim);
        float const lambda = testCase.headDim == 64 ? 0.003F : 0.001F;
        runLlmPagedMatchesContiguousCase<half>(testCase.batchSize, testCase.seqLen, testCase.numQHeads,
            testCase.numKVHeads, testCase.headDim, /*tokensPerPage=*/128, DataType::kHALF,
            /*slidingWindowSize=*/INT_MAX,
            /*fp8Input=*/false, /*qScale=*/1.0F, /*kScale=*/1.0F, /*vScale=*/1.0F, std::log2(lambda));
    }
}

TEST(CuteDslFMHARunnerTest, llmPagedKVFp8MatchesContiguous)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    constexpr float qScale = 0.01429094560444355F;
    constexpr float kScale = 0.021F;
    constexpr float vScale = 0.017F;
    runLlmPagedMatchesContiguousCase<__nv_fp8_e4m3>(
        1, 256, 12, 4, 128, 128, DataType::kFP8, INT_MAX, true, qScale, kScale, vScale);
    runLlmPagedMatchesContiguousCase<__nv_fp8_e4m3>(
        1, 256, 16, 2, 256, 128, DataType::kFP8, INT_MAX, true, qScale, kScale, vScale);
    runLlmPagedMatchesContiguousCase<__nv_fp8_e4m3>(
        1, 256, 16, 2, 256, 128, DataType::kFP8, 192, true, qScale, kScale, vScale);
}

TEST(CuteDslFMHARunnerTest, llmFp8LongSequenceAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    runLlmFp8LongSequenceAccuracyCase(32, 2, 128);
    runLlmFp8LongSequenceAccuracyCase(16, 2, 256);
}

// ==== ViT FP8 (input FP8 E4M3, output FP16) ====

#if SUPPORTS_FP8
// Per-element check at the same tolerance fmha.py's Python validation uses (atol=0.1, rtol=1e-5, see
// fmha.py:3242-3245).
void expectHalfOutputsCloseAtol(
    rt::Tensor const& actualTensor, rt::Tensor const& expectedTensor, std::string const& label, float atol, float rtol)
{
    ASSERT_EQ(actualTensor.getShape().volume(), expectedTensor.getShape().volume()) << label;

    auto const actual = copyDeviceToHost<half>(actualTensor);
    auto const expected = copyDeviceToHost<half>(expectedTensor);
    auto const& shape = actualTensor.getShape();

    bool nanDetected = false;
    int64_t const totalElements = static_cast<int64_t>(actual.size());

    for (int64_t idx = 0; idx < totalElements; ++idx)
    {
        float const actualValue = __half2float(actual[static_cast<size_t>(idx)]);
        float const expectedValue = __half2float(expected[static_cast<size_t>(idx)]);

        ASSERT_TRUE(isclose(actual[static_cast<size_t>(idx)], expected[static_cast<size_t>(idx)], rtol, atol))
            << label << " mismatch at index=" << formatTensorIndex(shape, idx) << " flat_index=" << idx
            << " expected=" << expectedValue << " actual=" << actualValue;

        nanDetected = nanDetected || std::isnan(actualValue);
    }
    EXPECT_FALSE(nanDetected) << label;
}

void runViTFp8AccuracyCase(std::vector<int32_t> const& cuSeqLens, int32_t numHeads, int32_t headDim, int32_t maxSeqLen)
{
    int32_t const batchSize = static_cast<int32_t>(cuSeqLens.size()) - 1;
    int32_t const totalSeqLen = cuSeqLens.back();

    size_t const qkvSize = static_cast<size_t>(totalSeqLen) * numHeads * headDim;

    // FP16 source data in [-1, 1]; per-tensor dequant scales picked so values fit
    // comfortably in FP8 E4M3 range (max ≈ 448).
    std::vector<half> qSourceFp16(qkvSize);
    std::vector<half> kSourceFp16(qkvSize);
    std::vector<half> vSourceFp16(qkvSize);
    uniformFloatInitialization(qSourceFp16, -1.0f, 1.0f);
    uniformFloatInitialization(kSourceFp16, -1.0f, 1.0f);
    uniformFloatInitialization(vSourceFp16, -1.0f, 1.0f);

    float const qScale = 0.05f;
    float const kScale = 0.05f;
    float const vScale = 0.05f;

    // Quantize once; reference and kernel both consume the same FP8-rounded values
    // (FP16 reference reads the dequantized form; FP8 kernel reads the raw FP8).
    auto const qFp8 = quantizeHalfToFp8(qSourceFp16, qScale);
    auto const kFp8 = quantizeHalfToFp8(kSourceFp16, kScale);
    auto const vFp8 = quantizeHalfToFp8(vSourceFp16, vScale);
    auto const qDqFp16 = dequantizeFp8ToHalf(qFp8, qScale);
    auto const kDqFp16 = dequantizeFp8ToHalf(kFp8, kScale);
    auto const vDqFp16 = dequantizeFp8ToHalf(vFp8, vScale);

    rt::Tensor qFp8Tensor({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kFP8);
    rt::Tensor kFp8Tensor({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kFP8);
    rt::Tensor vFp8Tensor({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kFP8);
    rt::Tensor qDqFp16Tensor({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kDqFp16Tensor({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vDqFp16Tensor({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputReference({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputCuteDsl({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(qFp8Tensor, qFp8);
    copyHostToDevice(kFp8Tensor, kFp8);
    copyHostToDevice(vFp8Tensor, vFp8);
    copyHostToDevice(qDqFp16Tensor, qDqFp16);
    copyHostToDevice(kDqFp16Tensor, kDqFp16);
    copyHostToDevice(vDqFp16Tensor, vDqFp16);
    copyHostToDevice(cuSeqLensTensor, cuSeqLens);

    cudaStream_t stream = nullptr;

    rt::launchFmhaReferenceCompact(qDqFp16Tensor, kDqFp16Tensor, vDqFp16Tensor, outputReference, cuSeqLensTensor,
        maxSeqLen, false, 1.0F / std::sqrt(static_cast<float>(headDim)), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    CuteDslFMHARunner runner(numHeads, numHeads, headDim);
    runner.run(qFp8Tensor.rawPointer(), kFp8Tensor.rawPointer(), vFp8Tensor.rawPointer(),
        outputCuteDsl.dataPointer<half>(), cuSeqLensTensor.dataPointer<int32_t>(), totalSeqLen, maxSeqLen, batchSize,
        stream, /*attentionScale=*/1.0F / std::sqrt(static_cast<float>(headDim)), /*fp8Input=*/true, qScale, kScale,
        vScale);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    expectHalfOutputsCloseAtol(outputCuteDsl, outputReference,
        "ViT FP8 CuTe DSL FMHA headDim=" + std::to_string(headDim) + " numHeads=" + std::to_string(numHeads),
        /*atol=*/0.1f, /*rtol=*/1e-5f);
}

//! d=72 contract case (Phi-4 SigLIP): FP8 cannot TMA-load 72-byte rows, so
//! callers zero-pad Q/K/V to d=80 and pass the real 1/sqrt(72) softmax
//! scale. This must be numerically EXACT vs true d=72 attention: zero
//! columns do not change the QK dots, and V zero columns produce zero O
//! columns. Reference runs at native d=72; the FP8 kernel runs at d=80 on
//! padded inputs; the first 72 output columns must match and the padded 8
//! must be zero.
void runViTFp8PaddedHeadDimCase(
    std::vector<int32_t> const& cuSeqLens, int32_t numHeads, int32_t realHeadDim, int32_t maxSeqLen)
{
    int32_t const paddedHeadDim = 80;
    int32_t const batchSize = static_cast<int32_t>(cuSeqLens.size()) - 1;
    int32_t const totalSeqLen = cuSeqLens.back();
    float const softmaxScale = 1.0F / std::sqrt(static_cast<float>(realHeadDim));

    size_t const realSize = static_cast<size_t>(totalSeqLen) * numHeads * realHeadDim;

    std::vector<half> qSourceFp16(realSize);
    std::vector<half> kSourceFp16(realSize);
    std::vector<half> vSourceFp16(realSize);
    uniformFloatInitialization(qSourceFp16, -1.0f, 1.0f);
    uniformFloatInitialization(kSourceFp16, -1.0f, 1.0f);
    uniformFloatInitialization(vSourceFp16, -1.0f, 1.0f);

    float const qScale = 0.05f;
    float const kScale = 0.05f;
    float const vScale = 0.05f;

    auto const qFp8 = quantizeHalfToFp8(qSourceFp16, qScale);
    auto const kFp8 = quantizeHalfToFp8(kSourceFp16, kScale);
    auto const vFp8 = quantizeHalfToFp8(vSourceFp16, vScale);
    auto const qDqFp16 = dequantizeFp8ToHalf(qFp8, qScale);
    auto const kDqFp16 = dequantizeFp8ToHalf(kFp8, kScale);
    auto const vDqFp16 = dequantizeFp8ToHalf(vFp8, vScale);

    // Zero-pad the FP8-rounded rows from realHeadDim to paddedHeadDim
    // (FP8 zero encodes exactly, matching the zero-padded weight rows the
    // export produces).
    auto padRows = [&](std::vector<__nv_fp8_e4m3> const& src) {
        std::vector<__nv_fp8_e4m3> dst(
            static_cast<size_t>(totalSeqLen) * numHeads * paddedHeadDim, __nv_fp8_e4m3(0.0f));
        for (int64_t row = 0; row < static_cast<int64_t>(totalSeqLen) * numHeads; ++row)
        {
            std::copy(src.begin() + row * realHeadDim, src.begin() + (row + 1) * realHeadDim,
                dst.begin() + row * paddedHeadDim);
        }
        return dst;
    };
    auto const qFp8Padded = padRows(qFp8);
    auto const kFp8Padded = padRows(kFp8);
    auto const vFp8Padded = padRows(vFp8);

    rt::Tensor qFp8Tensor({totalSeqLen, numHeads, paddedHeadDim}, rt::DeviceType::kGPU, DataType::kFP8);
    rt::Tensor kFp8Tensor({totalSeqLen, numHeads, paddedHeadDim}, rt::DeviceType::kGPU, DataType::kFP8);
    rt::Tensor vFp8Tensor({totalSeqLen, numHeads, paddedHeadDim}, rt::DeviceType::kGPU, DataType::kFP8);
    rt::Tensor qDqFp16Tensor({totalSeqLen, numHeads, realHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kDqFp16Tensor({totalSeqLen, numHeads, realHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vDqFp16Tensor({totalSeqLen, numHeads, realHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputReference({totalSeqLen, numHeads, realHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputCuteDsl({totalSeqLen, numHeads, paddedHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(qFp8Tensor, qFp8Padded);
    copyHostToDevice(kFp8Tensor, kFp8Padded);
    copyHostToDevice(vFp8Tensor, vFp8Padded);
    copyHostToDevice(qDqFp16Tensor, qDqFp16);
    copyHostToDevice(kDqFp16Tensor, kDqFp16);
    copyHostToDevice(vDqFp16Tensor, vDqFp16);
    copyHostToDevice(cuSeqLensTensor, cuSeqLens);

    cudaStream_t stream = nullptr;

    // Reference: true d=72 attention on the same FP8-rounded values.
    rt::launchFmhaReferenceCompact(qDqFp16Tensor, kDqFp16Tensor, vDqFp16Tensor, outputReference, cuSeqLensTensor,
        maxSeqLen, false, softmaxScale, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Kernel: d=80 FP8 variant on the padded inputs with the d=72 scale.
    CuteDslFMHARunner runner(numHeads, numHeads, paddedHeadDim);
    runner.run(qFp8Tensor.rawPointer(), kFp8Tensor.rawPointer(), vFp8Tensor.rawPointer(),
        outputCuteDsl.dataPointer<half>(), cuSeqLensTensor.dataPointer<int32_t>(), totalSeqLen, maxSeqLen, batchSize,
        stream, softmaxScale, /*fp8Input=*/true, qScale, kScale, vScale);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Compare the first realHeadDim columns against the true-d72 reference and
    // require the padded columns to be exactly zero.
    auto const actual = copyDeviceToHost<half>(outputCuteDsl);
    auto const expected = copyDeviceToHost<half>(outputReference);
    float maxAbsErr = 0.0f;
    float maxPadAbs = 0.0f;
    for (int64_t row = 0; row < static_cast<int64_t>(totalSeqLen) * numHeads; ++row)
    {
        for (int32_t c = 0; c < realHeadDim; ++c)
        {
            float const a = __half2float(actual[row * paddedHeadDim + c]);
            float const e = __half2float(expected[row * realHeadDim + c]);
            maxAbsErr = std::max(maxAbsErr, std::abs(a - e));
        }
        for (int32_t c = realHeadDim; c < paddedHeadDim; ++c)
        {
            maxPadAbs = std::max(maxPadAbs, std::abs(__half2float(actual[row * paddedHeadDim + c])));
        }
    }
    EXPECT_LE(maxAbsErr, 0.1f) << "d=" << realHeadDim << " contract: first " << realHeadDim
                               << " columns diverge from the native reference";
    EXPECT_EQ(maxPadAbs, 0.0f) << "d=" << realHeadDim << " contract: padded columns must be exactly zero";
}

#endif // SUPPORTS_FP8

#if SUPPORTS_FP8
TEST(CuteDslFMHARunnerTest, vitFp8Accuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    struct ViTFp8Case
    {
        std::vector<int32_t> cuSeqLens;
        int32_t numHeads;
        int32_t headDim;
        int32_t maxSeqLen;
    };

    // d=80 pads the MMA tiler K to 96 inside the kernel. d=72 has no direct
    // FP8 kernel (TMA needs 16B-aligned strides; 72 FP8 bytes are not) —
    // d=72 callers zero-pad to d=80 and pass the real softmax scale.
    std::vector<ViTFp8Case> const cases{
        {{0, 32, 60, 88, 128}, 14, 64, 128},
        {{0, 48, 120, 196}, 16, 80, 196},
        {{0, 40, 96, 200}, 16, 96, 256},
        {{0, 100, 200, 300}, 14, 128, 512},
    };

    for (auto const& testCase : cases)
    {
        std::string cuSeqLensStr = "[";
        for (size_t i = 0; i < testCase.cuSeqLens.size(); ++i)
        {
            if (i)
                cuSeqLensStr += ",";
            cuSeqLensStr += std::to_string(testCase.cuSeqLens[i]);
        }
        cuSeqLensStr += "]";
        SCOPED_TRACE(::testing::Message() << "numHeads=" << testCase.numHeads << " headDim=" << testCase.headDim
                                          << " maxSeqLen=" << testCase.maxSeqLen << " cuSeqLens=" << cuSeqLensStr);
        runViTFp8AccuracyCase(testCase.cuSeqLens, testCase.numHeads, testCase.headDim, testCase.maxSeqLen);
    }
}

TEST(CuteDslFMHARunnerTest, vitFp8PaddedHeadDim72)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }
    // Phi-4 SigLIP contract: zero-pad d=72 to d=80 + real 1/sqrt(72) scale
    // must reproduce true d=72 attention through the FP8 d=80 kernel.
    runViTFp8PaddedHeadDimCase({0, 48, 120, 196}, 16, 72, 196);
    runViTFp8PaddedHeadDimCase({0, 100, 200, 300}, 16, 72, 512);
}
#endif // SUPPORTS_FP8

#endif
