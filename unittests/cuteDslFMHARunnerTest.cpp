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

#ifdef CUTE_DSL_FMHA_ENABLED

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <climits>
#include <cmath>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "common/cudaUtils.h"
#include "contextAttnReference.h"
#include "kernels/contextAttentionKernels/cuteDslFMHARunner.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
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
    runner.run(qTensor.dataPointer<half>(), kTensor.dataPointer<half>(), vTensor.dataPointer<half>(),
        outputCuteDsl.dataPointer<half>(), cuSeqLensTensor.dataPointer<int32_t>(), totalSeqLen, maxSeqLen, batchSize,
        stream, attentionScale);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    expectHalfOutputsClose(outputCuteDsl, outputReference,
        "ViT CuTe DSL FMHA headDim=" + std::to_string(headDim) + " numHeads=" + std::to_string(numHeads));
}

void runLlmAccuracyCase(int32_t batchSize, int32_t seqLen, int32_t numQHeads, int32_t numKVHeads, int32_t headDim,
    float attentionScale, bool enableSkipSoftmax = false)
{
    size_t const qSize = static_cast<size_t>(batchSize) * seqLen * numQHeads * headDim;
    size_t const kvSize = static_cast<size_t>(batchSize) * seqLen * numKVHeads * headDim;

    std::vector<half> qInput(qSize);
    std::vector<half> kInput(kvSize);
    std::vector<half> vInput(kvSize);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);

    rt::Tensor qCute({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kCute({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vCute({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor qReference({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kReference({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vReference({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice(qCute, qInput);
    copyHostToDevice(kCute, kInput);
    copyHostToDevice(vCute, vInput);
    copyHostToDevice(qReference, qInput);
    copyHostToDevice(kReference, kInput);
    copyHostToDevice(vReference, vInput);

    rt::Tensor kvCacheCute({batchSize, 2, numKVHeads, seqLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvCacheReference({batchSize, 2, numKVHeads, seqLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputReference({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputCuteDsl({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cosSinCache({1, seqLen, headDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor kvCacheEndLens({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLens({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    CUDA_CHECK(cudaMemset(kvCacheCute.rawPointer(), 0, kvCacheCute.getShape().volume() * sizeof(half)));
    CUDA_CHECK(cudaMemset(kvCacheReference.rawPointer(), 0, kvCacheReference.getShape().volume() * sizeof(half)));
    CUDA_CHECK(cudaMemset(outputReference.rawPointer(), 0, outputReference.getShape().volume() * sizeof(half)));
    CUDA_CHECK(cudaMemset(outputCuteDsl.rawPointer(), 0, outputCuteDsl.getShape().volume() * sizeof(half)));

    std::vector<int32_t> kvCacheEndLensHost(static_cast<size_t>(batchSize), seqLen);
    std::vector<int32_t> cuKVSeqLensHost(static_cast<size_t>(batchSize + 1));
    for (int32_t idx = 0; idx <= batchSize; ++idx)
    {
        cuKVSeqLensHost[static_cast<size_t>(idx)] = idx * seqLen;
    }

    copyHostToDevice(kvCacheEndLens, kvCacheEndLensHost);
    copyHostToDevice(cuKVSeqLens, cuKVSeqLensHost);

    cudaStream_t stream = nullptr;
    std::vector<float> cosSinCacheHost(static_cast<size_t>(cosSinCache.getShape().volume()));
    uniformFloatInitialization(cosSinCacheHost, -1.0f, 1.0f);
    copyHostToDevice(cosSinCache, cosSinCacheHost);

    kernel::launchApplyRopeWriteKVSplitQKV(cosSinCache, kvCacheEndLens, qCute, kCute, vCute, kvCacheCute, 1.0f, 1.0f,
        stream, /*pageTable=*/nullptr, /*maxPagesPerSeq=*/0);
    kernel::launchApplyRopeWriteKV(cosSinCache, std::nullopt, qReference, kReference, vReference, kvCacheReference,
        1.0f, 1.0f, stream, true, /*pageTable=*/nullptr, /*maxPagesPerSeq=*/0);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    CuteDslFMHARunner runner(numQHeads, numKVHeads, headDim, batchSize, seqLen, seqLen);
    runner.run(qCute.dataPointer<half>(), kvCacheCute.dataPointer<half>(), outputCuteDsl.dataPointer<half>(),
        cuKVSeqLens.dataPointer<int32_t>(), stream, attentionScale, INT_MAX, false, 1.0F, 1.0F, 1.0F,
        enableSkipSoftmax);

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

template <typename T>
void runLlmPagedMatchesContiguousCase(int32_t batchSize, int32_t seqLen, int32_t numQHeads, int32_t numKVHeads,
    int32_t headDim, int32_t tokensPerPage, DataType dataType, int32_t slidingWindowSize = INT_MAX,
    bool fp8Input = false, float qScale = 1.0F, float kScale = 1.0F, float vScale = 1.0F)
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
    runner.run(qContiguous.rawPointer(), kvContiguousTensor.rawPointer(), outputContiguous.dataPointer<half>(),
        cuKVSeqLens.dataPointer<int32_t>(), stream, attentionScale, slidingWindowSize, fp8Input, qScale, kScale,
        vScale);
    runner.runPaged(qPaged.rawPointer(), kvPagedTensor.rawPointer(), pageListTensor.dataPointer<int32_t>(),
        outputPaged.dataPointer<half>(), cuKVSeqLens.dataPointer<int32_t>(), numPages, maxPagesPerSeq, tokensPerPage,
        dataType, stream, attentionScale, slidingWindowSize, fp8Input, qScale, kScale, vScale);
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
    runner.run(qFp8.rawPointer(), kvCacheFp8.rawPointer(), outputCuteDsl.rawPointer(),
        cuKVSeqLens.dataPointer<int32_t>(), stream, attentionScale, INT_MAX, true, qScale, kScale, vScale);
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
    EXPECT_TRUE(CuteDslFMHARunner::canImplement(128, 110));
    EXPECT_TRUE(CuteDslFMHARunner::canImplement(256, 110));
    EXPECT_FALSE(CuteDslFMHARunner::canImplement(256, 90));
}

TEST(CuteDslFMHARunnerTest, vitAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    if (!CuteDslFMHARunner::loadViTKernelModule())
    {
        GTEST_SKIP() << "Failed to load CuTe DSL ViT FMHA kernel module";
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

    if (!CuteDslFMHARunner::loadLLMKernelModule())
    {
        GTEST_SKIP() << "Failed to load CuTe DSL LLM FMHA kernel module";
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

    if (!CuteDslFMHARunner::loadLLMKernelModule())
    {
        GTEST_SKIP() << "Failed to load CuTe DSL LLM FMHA kernel module";
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
        runLlmAccuracyCase(testCase.batchSize, testCase.seqLen, testCase.numQHeads, testCase.numKVHeads,
            testCase.headDim, attentionScale, /*enableSkipSoftmax=*/true);
    }
}

TEST(CuteDslFMHARunnerTest, llmPagedKVMatchesContiguous)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    if (!CuteDslFMHARunner::loadLLMKernelModule())
    {
        FAIL() << "Failed to load CuTe DSL LLM FMHA kernel module";
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

TEST(CuteDslFMHARunnerTest, llmPagedKVFp8MatchesContiguous)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    if (!CuteDslFMHARunner::loadLLMKernelModule())
    {
        FAIL() << "Failed to load CuTe DSL LLM FMHA kernel module";
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

    if (!CuteDslFMHARunner::loadLLMKernelModule())
    {
        FAIL() << "Failed to load CuTe DSL LLM FMHA kernel module";
    }

    runLlmFp8LongSequenceAccuracyCase(32, 2, 128);
    runLlmFp8LongSequenceAccuracyCase(16, 2, 256);
}

#endif
