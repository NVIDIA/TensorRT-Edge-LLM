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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "contextAttnReference.h"
#include "kernels/contextAttentionKernels/cuteDslFMHAV2Runner.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "testUtils.h"

#if defined(CUTE_DSL_FMHA_BLACKWELL_ENABLED)
#include "kernels/contextAttentionKernels/cuteDslFMHARunner.h"
#include <cuda_fp8.h>
#endif

using namespace trt_edgellm;

void TestContextAttentionAccuracy(std::vector<int32_t> const& cuSeqlens, int32_t numQHeads, int32_t numKVHeads,
    int32_t headSize, int32_t maxSeqLen, bool isPackedViT = false, bool causal = true,
    std::optional<float> attentionScale = std::nullopt)
{
    float const resolvedAttentionScale = attentionScale.value_or(1.0F / std::sqrt(static_cast<float>(headSize)));
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    bool const canImplement = isPackedViT ? CuteDslFMHAV2Runner::canImplementViT(headSize, smVersion, DataType::kHALF)
                                          : causal
            && CuteDslFMHAV2Runner::canImplement(
                numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL);
    ASSERT_TRUE(canImplement) << "FMHA-v2 CuTe DSL unexpectedly unsupported for headSize=" << headSize
                              << ", SM=" << smVersion;

    // Calculate total elements
    int32_t const batchSize = static_cast<int32_t>(cuSeqlens.size()) - 1;
    int32_t const totalTokens = isPackedViT ? cuSeqlens.back() : batchSize * maxSeqLen;

    size_t const qSize = static_cast<size_t>(totalTokens) * numQHeads * headSize;
    size_t const kvSize = static_cast<size_t>(totalTokens) * numKVHeads * headSize;
    size_t const outSize = static_cast<size_t>(totalTokens) * numQHeads * headSize;

    // Initialize input data
    std::vector<half> qInput(qSize);
    std::vector<half> kInput(kvSize);
    std::vector<half> vInput(kvSize);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);

    // Create Tensor objects based on layout (they allocate device memory internally)
    rt::Tensor qTensor, kTensor, vTensor, oTensorRef, oTensorKernel;
    if (isPackedViT)
    {
        qTensor = rt::Tensor({totalTokens, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        kTensor = rt::Tensor({totalTokens, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        vTensor = rt::Tensor({totalTokens, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        oTensorRef = rt::Tensor({totalTokens, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        oTensorKernel = rt::Tensor({totalTokens, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    }
    else
    {
        qTensor = rt::Tensor({batchSize, maxSeqLen, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        kTensor = rt::Tensor({batchSize, maxSeqLen, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        vTensor = rt::Tensor({batchSize, maxSeqLen, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        oTensorRef = rt::Tensor({batchSize, maxSeqLen, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
        oTensorKernel = rt::Tensor({batchSize, maxSeqLen, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    }

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(qTensor.rawPointer(), qInput.data(), qSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kTensor.rawPointer(), kInput.data(), kvSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vTensor.rawPointer(), vInput.data(), kvSize * sizeof(half), cudaMemcpyHostToDevice));
    rt::Tensor cuSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    CUDA_CHECK(cudaMemcpy(
        cuSeqLensTensor.rawPointer(), cuSeqlens.data(), (batchSize + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    rt::Tensor paddedCuKVSeqLensTensor;
    if (!isPackedViT)
    {
        std::vector<int32_t> inputSeqLens(batchSize);
        for (int32_t i = 0; i < batchSize; ++i)
        {
            inputSeqLens[i] = cuSeqlens[i + 1] - cuSeqlens[i];
        }

        rt::Tensor inputSeqLensTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        CUDA_CHECK(cudaMemcpy(
            inputSeqLensTensor.rawPointer(), inputSeqLens.data(), batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));
        rt::Tensor cuQSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
        rt::Tensor cuKVSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
        rt::Tensor kvCacheEndIdxsTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        paddedCuKVSeqLensTensor = rt::Tensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
        kernel::calCuQCuKVSeqLensAndKVEndIdxs(inputSeqLensTensor, rt::Tensor{}, cuQSeqLensTensor, cuKVSeqLensTensor,
            kvCacheEndIdxsTensor, paddedCuKVSeqLensTensor, maxSeqLen, nullptr);
    }

    // Compute reference output
    cudaStream_t stream = nullptr;
    if (isPackedViT)
    {
        rt::launchFmhaReferenceCompact(
            qTensor, kTensor, vTensor, oTensorRef, cuSeqLensTensor, maxSeqLen, false, resolvedAttentionScale, stream);
    }
    else
    {
        rt::launchFmhaReferenceBshd(qTensor, kTensor, vTensor, oTensorRef, causal, resolvedAttentionScale, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Copy reference output to host
    std::vector<half> outReference(outSize);
    CUDA_CHECK(
        cudaMemcpy(outReference.data(), oTensorRef.rawPointer(), outSize * sizeof(half), cudaMemcpyDeviceToHost));

    CuteDslFMHAV2Runner runner(numQHeads, numKVHeads, headSize, batchSize, maxSeqLen, maxSeqLen,
        /*useSmallD64=*/maxSeqLen <= 512);
    bool ranKernel = false;
    if (isPackedViT)
    {
        ranKernel = runner.run(qTensor.rawPointer(), kTensor.rawPointer(), vTensor.rawPointer(),
            oTensorKernel.rawPointer(), cuSeqLensTensor.dataPointer<int32_t>(), totalTokens, maxSeqLen, batchSize,
            stream, resolvedAttentionScale);
    }
    else
    {
        ranKernel = runner.run(qTensor.rawPointer(), kTensor.rawPointer(), vTensor.rawPointer(),
            oTensorKernel.rawPointer(), paddedCuKVSeqLensTensor.dataPointer<int32_t>(), stream, resolvedAttentionScale);
    }
    ASSERT_TRUE(ranKernel);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Copy output to host
    std::vector<half> outHost(outSize);
    CUDA_CHECK(cudaMemcpy(outHost.data(), oTensorKernel.rawPointer(), outSize * sizeof(half), cudaMemcpyDeviceToHost));

    // Check accuracy
    bool NanValueDetected = false;
    int32_t numCloseWithin1E_3 = 0;
    int64_t totalElements = static_cast<int64_t>(outSize);

    for (int64_t i = 0; i < totalElements; ++i)
    {
        ASSERT_TRUE(isclose(outHost[i], outReference[i], 1e-2, 1e-2))
            << "Mismatch at index=" << i << " expected=" << __half2float(outReference[i])
            << " actual=" << __half2float(outHost[i]);

        if (isclose(outHost[i], outReference[i], 1e-3, 1e-3))
        {
            numCloseWithin1E_3++;
        }
        if (std::isnan(__half2float(outHost[i])))
        {
            NanValueDetected = true;
        }
    }

    float passRate1E_3 = static_cast<float>(numCloseWithin1E_3) / totalElements;

    std::string layoutStr = isPackedViT ? "[Compact]" : "[Padded]";
    std::string maskStr = causal ? "[Causal] " : "[Non-causal] ";
    std::cout << "Context Attention test. " << layoutStr << maskStr << "batch_size: " << batchSize;
    if (isPackedViT)
    {
        std::cout << " total_tokens: " << totalTokens << " max_seq_len: " << maxSeqLen;
    }
    else
    {
        std::cout << " seq_len: " << maxSeqLen;
    }
    std::cout << " num_Q_heads: " << numQHeads << " num_KV_heads: " << numKVHeads << " head_size: " << headSize
              << " pass_rate_1e-3: " << passRate1E_3 << std::endl;

    EXPECT_GT(passRate1E_3, 0.9);
    EXPECT_FALSE(NanValueDetected);
}

// Convenience wrapper for padded layout (fixed sequence length)
void TestContextAttentionAccuracy(int32_t batchSize, int32_t seqLen, int32_t numQHeads, int32_t numKVHeads,
    int32_t headSize, bool causal = true, std::optional<float> attentionScale = std::nullopt)
{
    // Generate cu_seqlens for fixed-length sequences
    std::vector<int32_t> cuSeqlens(batchSize + 1);
    for (int32_t i = 0; i <= batchSize; i++)
    {
        cuSeqlens[i] = i * seqLen;
    }
    TestContextAttentionAccuracy(cuSeqlens, numQHeads, numKVHeads, headSize, seqLen, false, causal, attentionScale);
}

namespace
{

size_t pagedV2BshdIndex(
    int32_t batch, int32_t seq, int32_t head, int32_t dim, int32_t seqLen, int32_t numHeads, int32_t headDim)
{
    return static_cast<size_t>(((static_cast<int64_t>(batch) * seqLen + seq) * numHeads + head) * headDim + dim);
}

size_t pagedV2PoolIndex(int32_t page, int32_t token, int32_t head, int32_t dim, int32_t numHeads, int32_t headDim)
{
    int32_t constexpr kTOKENS_PER_PAGE = 128;
    return static_cast<size_t>(
        ((static_cast<int64_t>(page) * kTOKENS_PER_PAGE + token) * numHeads + head) * headDim + dim);
}

std::vector<float> computePagedV2Reference(std::vector<half> const& q, std::vector<half> const& k,
    std::vector<half> const& v, int32_t batchSize, int32_t seqLenQ, int32_t seqLenKCapacity,
    std::vector<int32_t> const& seqLensQ, std::vector<int32_t> const& seqLensK, int32_t numQHeads, int32_t numKVHeads,
    int32_t headDim, float attentionScale, int32_t windowSizeLeft, std::vector<float> const& attentionSinks = {},
    std::vector<float>* sinkMassFractions = nullptr)
{
    std::vector<float> output(static_cast<size_t>(batchSize) * seqLenQ * numQHeads * headDim);
    if (sinkMassFractions != nullptr)
    {
        sinkMassFractions->resize(static_cast<size_t>(batchSize) * seqLenQ * numQHeads);
    }
    int32_t const groupSize = numQHeads / numKVHeads;
    std::vector<float> weights(static_cast<size_t>(seqLenKCapacity));
    std::vector<float> scores(static_cast<size_t>(seqLenKCapacity));

    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        int32_t const actualSeqLenQ = seqLensQ[static_cast<size_t>(batch)];
        int32_t const seqLenK = seqLensK[static_cast<size_t>(batch)];
        int32_t const qOffset = seqLenK - actualSeqLenQ;
        for (int32_t qPos = 0; qPos < actualSeqLenQ; ++qPos)
        {
            int32_t const keyEnd = qOffset + qPos;
            int32_t const keyBegin = windowSizeLeft == INT_MAX ? 0 : std::max(0, keyEnd - windowSizeLeft);
            for (int32_t qHead = 0; qHead < numQHeads; ++qHead)
            {
                int32_t const kvHead = qHead / groupSize;
                float rowMax = attentionSinks.empty() ? -std::numeric_limits<float>::infinity()
                                                      : attentionSinks[static_cast<size_t>(qHead)];
                for (int32_t kPos = keyBegin; kPos <= keyEnd; ++kPos)
                {
                    float score = 0.0F;
                    for (int32_t dim = 0; dim < headDim; ++dim)
                    {
                        size_t const qIdx = pagedV2BshdIndex(batch, qPos, qHead, dim, seqLenQ, numQHeads, headDim);
                        size_t const kIdx
                            = pagedV2BshdIndex(batch, kPos, kvHead, dim, seqLenKCapacity, numKVHeads, headDim);
                        score += __half2float(q[qIdx]) * __half2float(k[kIdx]);
                    }
                    scores[static_cast<size_t>(kPos)] = score;
                    rowMax = std::max(rowMax, score * attentionScale);
                }

                float rowSum
                    = attentionSinks.empty() ? 0.0F : std::exp(attentionSinks[static_cast<size_t>(qHead)] - rowMax);
                float attentionRowSum = 0.0F;
                for (int32_t kPos = keyBegin; kPos <= keyEnd; ++kPos)
                {
                    float const weight = std::exp(scores[static_cast<size_t>(kPos)] * attentionScale - rowMax);
                    weights[static_cast<size_t>(kPos)] = weight;
                    rowSum += weight;
                    attentionRowSum += weight;
                }
                if (sinkMassFractions != nullptr)
                {
                    (*sinkMassFractions)[(static_cast<size_t>(batch) * seqLenQ + qPos) * numQHeads + qHead]
                        = attentionRowSum / rowSum;
                }

                // kPos is the outer loop so V rows are read contiguously; each output element accumulates over kPos
                // in increasing order.
                std::vector<float> values(static_cast<size_t>(headDim), 0.0F);
                for (int32_t kPos = keyBegin; kPos <= keyEnd; ++kPos)
                {
                    for (int32_t dim = 0; dim < headDim; ++dim)
                    {
                        size_t const vIdx
                            = pagedV2BshdIndex(batch, kPos, kvHead, dim, seqLenKCapacity, numKVHeads, headDim);
                        values[static_cast<size_t>(dim)]
                            += weights[static_cast<size_t>(kPos)] * __half2float(v[vIdx]) / rowSum;
                    }
                }
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    output[pagedV2BshdIndex(batch, qPos, qHead, dim, seqLenQ, numQHeads, headDim)]
                        = values[static_cast<size_t>(dim)];
                }
            }
        }
    }
    return output;
}

//! @param skipSoftmaxLambda > 0 binds the one-row skip-softmax scale table {0, S = lambda * max seqLenK} (0 binds
//! none: the runner hands the kernel the dense {0, 0} row). K on the causal diagonal is then set to 8x the Q of each KV
//! group's first head so the diagonal tile fixes every row's running max and earlier tiles fall far below lambda: the
//! skip path fires while the skipped mass stays negligible.
//! @param skipSoftmaxExpectDense the bound table must reproduce the dense result on unstructured data (dense
//! tolerance): the predicate never trips (tiny lambda) or the in-kernel guard forces dense (S >= seqlen_kv).
void TestContextAttentionPagedAccuracy(int32_t headDim, int32_t numQHeads, int32_t numKVHeads, int32_t seqLenQ,
    int32_t seqLenKCapacity, int32_t windowSizeLeft = INT_MAX,
    std::optional<float> requestedAttentionScale = std::nullopt, std::vector<int32_t> seqLensQ = {},
    std::vector<int32_t> seqLensK = {}, bool nanPoison = false, bool packedQ = false, float skipSoftmaxLambda = 0.0F,
    bool skipSoftmaxExpectDense = false, bool optimizedBlackwell = false, bool fp8Input = false,
    std::vector<float> const& attentionSinks = {})
{
    int32_t constexpr kTOKENS_PER_PAGE = 128;
    int32_t const batchSize = seqLensQ.empty() ? (seqLensK.empty() ? 1 : static_cast<int32_t>(seqLensK.size()))
                                               : static_cast<int32_t>(seqLensQ.size());
    if (seqLensQ.empty())
    {
        seqLensQ.assign(static_cast<size_t>(batchSize), seqLenQ);
    }
    if (seqLensK.empty())
    {
        seqLensK.assign(static_cast<size_t>(batchSize), seqLenKCapacity);
    }
    ASSERT_GT(seqLenQ, 0);
    ASSERT_GT(batchSize, 0);
    ASSERT_EQ(seqLensQ.size(), seqLensK.size());
    ASSERT_EQ(numQHeads % numKVHeads, 0);
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        int32_t const actualSeqLenQ = seqLensQ[static_cast<size_t>(batch)];
        int32_t const seqLenK = seqLensK[static_cast<size_t>(batch)];
        ASSERT_GT(actualSeqLenQ, 0);
        ASSERT_LE(actualSeqLenQ, seqLenQ);
        ASSERT_GE(seqLenK, actualSeqLenQ);
        ASSERT_LE(seqLenK, seqLenKCapacity);
    }

    int32_t const livePagesPerSeq = (seqLenKCapacity + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE;
    // Keep one extra logical page-table slot poisoned. The kernel must stop from
    // cuKVSeqLens before resolving it.
    int32_t const maxPagesPerSeq = livePagesPerSeq + 1;
    int32_t const capacity = maxPagesPerSeq * kTOKENS_PER_PAGE;
    // Give every batch an independent scrambled physical-page range and leave
    // two entire pages per plane poisoned.
    int32_t const pagesPerPlane = batchSize * livePagesPerSeq + 2;
    int32_t const numFlatPages = 2 * pagesPerPlane;
    float const attentionScale = requestedAttentionScale.value_or(1.0F / std::sqrt(static_cast<float>(headDim)));

    size_t const qSize = static_cast<size_t>(batchSize) * seqLenQ * numQHeads * headDim;
    size_t const kvSize = static_cast<size_t>(batchSize) * seqLenKCapacity * numKVHeads * headDim;
    size_t const poolSize = static_cast<size_t>(numFlatPages) * kTOKENS_PER_PAGE * numKVHeads * headDim;

    std::vector<half> qHost(qSize);
    std::vector<half> kHost(kvSize);
    std::vector<half> vHost(kvSize);
    uniformFloatInitialization(qHost, -1.0F, 1.0F);
    uniformFloatInitialization(kHost, -1.0F, 1.0F);
    uniformFloatInitialization(vHost, -1.0F, 1.0F);
    [[maybe_unused]] float const qScale = fp8Input ? 0.25F : 1.0F;
    [[maybe_unused]] float const kScale = fp8Input ? 0.5F : 1.0F;
    [[maybe_unused]] float const vScale = fp8Input ? 2.0F : 1.0F;
    if (!attentionSinks.empty())
    {
        uniformFloatInitialization(vHost, 0.5F, 1.5F);
    }
#if defined(CUTE_DSL_FMHA_BLACKWELL_ENABLED)
    if (fp8Input)
    {
        auto roundInput = [](std::vector<half>& values, float scale) {
            for (auto& value : values)
            {
                value = __float2half(static_cast<float>(__nv_fp8_e4m3(__half2float(value) / scale)) * scale);
            }
        };
        roundInput(qHost, qScale);
        roundInput(kHost, kScale);
        roundInput(vHost, vScale);
    }
#endif
    bool const enableSkipSoftmax = skipSoftmaxLambda > 0.0F;
    bool const skipSoftmaxApproximate = enableSkipSoftmax && !skipSoftmaxExpectDense;
    if (skipSoftmaxApproximate)
    {
        int32_t const groupSize = numQHeads / numKVHeads;
        for (int32_t batch = 0; batch < batchSize; ++batch)
        {
            int32_t const offset = seqLensK[static_cast<size_t>(batch)] - seqLensQ[static_cast<size_t>(batch)];
            for (int32_t seq = 0; seq < seqLensQ[static_cast<size_t>(batch)]; ++seq)
            {
                for (int32_t kvHead = 0; kvHead < numKVHeads; ++kvHead)
                {
                    for (int32_t dim = 0; dim < headDim; ++dim)
                    {
                        size_t const qIdx
                            = pagedV2BshdIndex(batch, seq, kvHead * groupSize, dim, seqLenQ, numQHeads, headDim);
                        size_t const kIdx
                            = pagedV2BshdIndex(batch, seq + offset, kvHead, dim, seqLenKCapacity, numKVHeads, headDim);
                        kHost[kIdx] = __float2half(8.0F * __half2float(qHost[qIdx]));
                    }
                }
            }
        }
    }
    std::vector<int32_t> const skipSoftmaxScaleTable{0,
        static_cast<int32_t>(
            skipSoftmaxLambda * static_cast<float>(*std::max_element(seqLensK.begin(), seqLensK.end())))};

    // Padding embeddings can overflow FP16 and turn NaN from the first layer on, so the padded
    // rows/keys must not reach a valid row even as NaN/Inf.
    half const qPoison = nanPoison ? __float2half(std::numeric_limits<float>::quiet_NaN()) : __float2half(120.0F);
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        for (int32_t seq = seqLensQ[static_cast<size_t>(batch)]; seq < seqLenQ; ++seq)
        {
            for (int32_t head = 0; head < numQHeads; ++head)
            {
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    qHost[pagedV2BshdIndex(batch, seq, head, dim, seqLenQ, numQHeads, headDim)] = qPoison;
                }
            }
        }
    }

    half const kPoison = nanPoison ? __float2half(-std::numeric_limits<float>::infinity()) : __float2half(-120.0F);
    half const vPoison = nanPoison ? __float2half(std::numeric_limits<float>::quiet_NaN()) : __float2half(112.0F);
    std::vector<half> poolHost(poolSize, kPoison);
    for (int32_t page = pagesPerPlane; page < numFlatPages; ++page)
    {
        for (int32_t token = 0; token < kTOKENS_PER_PAGE; ++token)
        {
            for (int32_t head = 0; head < numKVHeads; ++head)
            {
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    size_t const poolIndex = pagedV2PoolIndex(page, token, head, dim, numKVHeads, headDim);
                    poolHost[poolIndex] = vPoison;
                }
            }
        }
    }

    std::vector<int32_t> pageList(static_cast<size_t>(batchSize) * 2 * maxPagesPerSeq, -1);
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        int32_t const batchLivePages = (seqLensK[static_cast<size_t>(batch)] + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE;
        for (int32_t logicalPage = 0; logicalPage < batchLivePages; ++logicalPage)
        {
            int32_t const kPage = 1 + batch * livePagesPerSeq + (livePagesPerSeq - logicalPage - 1);
            int32_t const vPage = pagesPerPlane + 1 + batch * livePagesPerSeq + ((logicalPage + 1) % livePagesPerSeq);
            pageList[static_cast<size_t>((batch * 2) * maxPagesPerSeq + logicalPage)] = kPage;
            pageList[static_cast<size_t>((batch * 2 + 1) * maxPagesPerSeq + logicalPage)] = vPage;
        }
    }

    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        for (int32_t seq = 0; seq < seqLensK[static_cast<size_t>(batch)]; ++seq)
        {
            int32_t const logicalPage = seq / kTOKENS_PER_PAGE;
            int32_t const tokenInPage = seq % kTOKENS_PER_PAGE;
            int32_t const kPage = pageList[static_cast<size_t>((batch * 2) * maxPagesPerSeq + logicalPage)];
            int32_t const vPage = pageList[static_cast<size_t>((batch * 2 + 1) * maxPagesPerSeq + logicalPage)];
            for (int32_t head = 0; head < numKVHeads; ++head)
            {
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    size_t const denseIndex
                        = pagedV2BshdIndex(batch, seq, head, dim, seqLenKCapacity, numKVHeads, headDim);
                    size_t const kPoolIndex = pagedV2PoolIndex(kPage, tokenInPage, head, dim, numKVHeads, headDim);
                    size_t const vPoolIndex = pagedV2PoolIndex(vPage, tokenInPage, head, dim, numKVHeads, headDim);
                    poolHost[kPoolIndex] = kHost[denseIndex];
                    poolHost[vPoolIndex] = vHost[denseIndex];
                }
            }
        }
    }

    int32_t totalQSeqLen = 0;
    for (int32_t const qSeqLen : seqLensQ)
    {
        totalQSeqLen += qSeqLen;
    }
    int32_t const maxQSeqLen = *std::max_element(seqLensQ.begin(), seqLensQ.end());
    std::vector<half> packedQHost;
    if (packedQ)
    {
        packedQHost.reserve(static_cast<size_t>(totalQSeqLen) * numQHeads * headDim);
        for (int32_t batch = 0; batch < batchSize; ++batch)
        {
            size_t const begin = pagedV2BshdIndex(batch, 0, 0, 0, seqLenQ, numQHeads, headDim);
            size_t const count = static_cast<size_t>(seqLensQ[static_cast<size_t>(batch)]) * numQHeads * headDim;
            packedQHost.insert(packedQHost.end(), qHost.begin() + begin, qHost.begin() + begin + count);
        }
    }
    rt::Tensor qTensor(
        packedQ ? rt::Coords{totalQSeqLen, numQHeads, headDim} : rt::Coords{batchSize, seqLenQ, numQHeads, headDim},
        rt::DeviceType::kGPU, fp8Input ? DataType::kFP8 : DataType::kHALF);
    rt::Tensor poolTensor({numFlatPages, kTOKENS_PER_PAGE, numKVHeads, headDim}, rt::DeviceType::kGPU,
        fp8Input ? DataType::kFP8 : DataType::kHALF);
    rt::Tensor pageListTensor({batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor outputPaged(
        packedQ ? rt::Coords{totalQSeqLen, numQHeads, headDim} : rt::Coords{batchSize, seqLenQ, numQHeads, headDim},
        rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuQSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor skipSoftmaxScaleTableTensor({2}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor attentionSinksTensor({numQHeads}, rt::DeviceType::kGPU, DataType::kFLOAT);
    std::vector<int32_t> cuQSeqLens(static_cast<size_t>(batchSize + 1), 0);
    std::vector<int32_t> cuKVSeqLens(static_cast<size_t>(batchSize + 1), 0);
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        cuQSeqLens[static_cast<size_t>(batch + 1)]
            = cuQSeqLens[static_cast<size_t>(batch)] + seqLensQ[static_cast<size_t>(batch)];
        cuKVSeqLens[static_cast<size_t>(batch + 1)]
            = cuKVSeqLens[static_cast<size_t>(batch)] + seqLensK[static_cast<size_t>(batch)];
    }
#if defined(CUTE_DSL_FMHA_BLACKWELL_ENABLED)
    if (fp8Input)
    {
        auto toFp8 = [](std::vector<half> const& values, float scale) {
            std::vector<__nv_fp8_e4m3> quantized(values.size());
            std::transform(values.begin(), values.end(), quantized.begin(),
                [scale](half value) { return __nv_fp8_e4m3(__half2float(value) / scale); });
            return quantized;
        };
        copyHostToDevice(qTensor, toFp8(packedQHost, qScale));
        std::vector<__nv_fp8_e4m3> quantizedPool(poolHost.size());
        size_t const planeSize = poolHost.size() / 2;
        for (size_t i = 0; i < poolHost.size(); ++i)
        {
            quantizedPool[i] = __nv_fp8_e4m3(__half2float(poolHost[i]) / (i < planeSize ? kScale : vScale));
        }
        copyHostToDevice(poolTensor, quantizedPool);
    }
    else
#endif
    {
        copyHostToDevice(qTensor, packedQ ? packedQHost : qHost);
        copyHostToDevice(poolTensor, poolHost);
    }
    rt::Tensor sinkTensor;
    if (optimizedBlackwell)
    {
        sinkTensor = rt::Tensor({numQHeads}, rt::DeviceType::kGPU, DataType::kFLOAT);
        if (!attentionSinks.empty())
        {
            copyHostToDevice(sinkTensor, attentionSinks);
        }
    }
    copyHostToDevice(pageListTensor, pageList);
    copyHostToDevice(cuQSeqLensTensor, cuQSeqLens);
    copyHostToDevice(cuKVSeqLensTensor, cuKVSeqLens);
    copyHostToDevice(skipSoftmaxScaleTableTensor, skipSoftmaxScaleTable);
    if (!attentionSinks.empty())
    {
        copyHostToDevice(attentionSinksTensor, attentionSinks);
    }

    CuteDslFMHAV2Runner runner(numQHeads, numKVHeads, headDim, batchSize, seqLenQ, capacity);
    if (enableSkipSoftmax)
    {
#if CUDA_VERSION < 12000
        GTEST_SKIP() << "The CUDA 11 FMHA-v2 artifact has no skip-softmax path.";
#endif
    }
    cudaStream_t stream = nullptr;
    if (optimizedBlackwell)
    {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    Defer const streamGuard([stream]() {
        if (stream != nullptr)
        {
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
    });
    auto const runPaged = [&](float const* sinkPtr) {
#if defined(CUTE_DSL_FMHA_BLACKWELL_ENABLED)
        if (optimizedBlackwell)
        {
            CuteDslFMHARunner blackwell(numQHeads, numKVHeads, headDim, batchSize, seqLenQ, capacity);
            if (!blackwell.preflightPagedRagged(stream, windowSizeLeft, fp8Input ? DataType::kFP8 : DataType::kHALF))
            {
                return false;
            }
            return blackwell.runPagedRagged(qTensor.rawPointer(), poolTensor.rawPointer(),
                pageListTensor.dataPointer<int32_t>(), outputPaged.rawPointer(),
                cuQSeqLensTensor.dataPointer<int32_t>(), cuKVSeqLensTensor.dataPointer<int32_t>(), totalQSeqLen,
                maxQSeqLen, numFlatPages, maxPagesPerSeq, kTOKENS_PER_PAGE, stream, attentionScale, windowSizeLeft,
                fp8Input ? DataType::kFP8 : DataType::kHALF, qScale, kScale, vScale, sinkPtr);
        }
#endif
        if (packedQ)
        {
            CuteDslFMHAV2Runner::PagedRaggedParams params{};
            params.qPtr = qTensor.rawPointer();
            params.pagedKVPoolPtr = poolTensor.rawPointer();
            params.kvCachePageList = pageListTensor.dataPointer<int32_t>();
            params.oPtr = outputPaged.rawPointer();
            params.cuQSeqLens = cuQSeqLensTensor.dataPointer<int32_t>();
            params.cuKVSeqLens = cuKVSeqLensTensor.dataPointer<int32_t>();
            params.totalQSeqLen = totalQSeqLen;
            params.maxQSeqLen = maxQSeqLen;
            params.numFlatPages = numFlatPages;
            params.maxPagesPerSeq = maxPagesPerSeq;
            params.tokensPerPage = kTOKENS_PER_PAGE;
            params.stream = stream;
            params.attentionScale = attentionScale;
            params.slidingWindowSize = windowSizeLeft;
            params.attentionSinks = attentionSinks.empty() ? nullptr : attentionSinksTensor.dataPointer<float>();
            return runner.runPagedRagged(params);
        }
        return runner.runPaged(qTensor.rawPointer(), poolTensor.rawPointer(), pageListTensor.dataPointer<int32_t>(),
            outputPaged.rawPointer(), cuQSeqLensTensor.dataPointer<int32_t>(), cuKVSeqLensTensor.dataPointer<int32_t>(),
            numFlatPages, maxPagesPerSeq, kTOKENS_PER_PAGE, stream, attentionScale, windowSizeLeft,
            enableSkipSoftmax ? skipSoftmaxScaleTableTensor.dataPointer<int32_t>() : nullptr,
            enableSkipSoftmax ? 2 : 0);
    };
    std::vector<half> fp8Baseline;
    if (optimizedBlackwell && fp8Input)
    {
        if (attentionSinks.empty())
        {
            copyHostToDevice(sinkTensor, std::vector<float>(numQHeads, -std::numeric_limits<float>::infinity()));
        }
        ASSERT_TRUE(runPaged(attentionSinks.empty() ? sinkTensor.dataPointer<float>() : nullptr));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fp8Baseline = copyDeviceToHost<half>(outputPaged);
    }
    ASSERT_TRUE(runPaged(optimizedBlackwell && !attentionSinks.empty() ? sinkTensor.dataPointer<float>() : nullptr));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    auto const actual = copyDeviceToHost<half>(outputPaged);
    std::vector<float> sinkMassFractions;
    auto const expected = computePagedV2Reference(qHost, kHost, vHost, batchSize, seqLenQ, seqLenKCapacity, seqLensQ,
        seqLensK, numQHeads, numKVHeads, headDim, attentionScale, windowSizeLeft, attentionSinks,
        optimizedBlackwell && fp8Input && !attentionSinks.empty() ? &sinkMassFractions : nullptr);
    int64_t closeWithin1e3 = 0;
    int64_t validElements = 0;
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        for (int32_t seq = 0; seq < seqLensQ[static_cast<size_t>(batch)]; ++seq)
        {
            for (int32_t head = 0; head < numQHeads; ++head)
            {
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    size_t const idx = pagedV2BshdIndex(batch, seq, head, dim, seqLenQ, numQHeads, headDim);
                    size_t const actualIdx = packedQ
                        ? (static_cast<size_t>(cuQSeqLens[static_cast<size_t>(batch)] + seq) * numQHeads + head)
                                * headDim
                            + dim
                        : idx;
                    float const actualValue = __half2float(actual[actualIdx]);
                    // Skip-softmax is approximate by design; its gate is the 0.1 max-abs the calibrated
                    // lambda is certified against, not the dense 1e-2 tolerance.
                    float expectedValue = expected[idx];
                    bool close;
                    if (optimizedBlackwell && fp8Input)
                    {
                        if (attentionSinks.empty())
                        {
                            expectedValue = __half2float(fp8Baseline[actualIdx]);
                            close = std::memcmp(&actual[actualIdx], &fp8Baseline[actualIdx], sizeof(half)) == 0;
                        }
                        else
                        {
                            expectedValue = __half2float(fp8Baseline[actualIdx]) * sinkMassFractions[idx / headDim];
                            close = isclose(actualValue, expectedValue, 1e-3F, 1e-3F);
                        }
                    }
                    else
                    {
                        close = skipSoftmaxApproximate ? std::fabs(actualValue - expectedValue) < 0.1F
                                                       : isclose(actualValue, expectedValue, 1e-2F, 1e-2F);
                    }
                    ASSERT_TRUE(close) << "Paged FMHA mismatch at index=" << idx << " batch=" << batch << " seq=" << seq
                                       << " D=" << headDim << " SqCapacity=" << seqLenQ
                                       << " actualSq=" << seqLensQ[static_cast<size_t>(batch)]
                                       << " SkCapacity=" << seqLenKCapacity
                                       << " actualSk=" << seqLensK[static_cast<size_t>(batch)]
                                       << " windowLeft=" << windowSizeLeft << " skipLambda=" << skipSoftmaxLambda
                                       << " expected=" << expectedValue << " actual=" << actualValue;
                    closeWithin1e3 += isclose(actualValue, expected[idx], 1e-3F, 1e-3F);
                    ASSERT_FALSE(std::isnan(actualValue));
                    ASSERT_FALSE(std::isinf(actualValue));
                    ++validElements;
                }
            }
        }
    }
    if (!skipSoftmaxApproximate && !fp8Input)
    {
        EXPECT_GT(static_cast<float>(closeWithin1e3) / static_cast<float>(validElements), 0.9F);
    }
}

void assertPagedCapability(int32_t headDim, CuteDslFMHAV2MaskType maskType)
{
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    ASSERT_TRUE(CuteDslFMHAV2Runner::canImplementPaged(8, 2, headDim, smVersion, DataType::kHALF, maskType))
        << "Native paged FMHA-v2 unexpectedly unsupported for D=" << headDim << ", SM=" << smVersion;
}

} // namespace

TEST(ContextAttentionTest, fmhaV2CapabilityContract)
{
#if !defined(CUTE_DSL_FMHA_ENABLED)
    EXPECT_FALSE(CuteDslFMHAV2Runner::canImplement(8, 2, 64, 100, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
    EXPECT_FALSE(
        CuteDslFMHAV2Runner::canImplementPaged(8, 2, 64, 100, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
    EXPECT_FALSE(
        CuteDslFMHAV2Runner::canImplementPagedRagged(8, 2, 64, 100, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
    return;
#endif

    std::vector<int32_t> const supportedSMs{80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121};
    std::vector<int32_t> const supportedHeadDims{64, 128, 256, 512};

    for (int32_t const smVersion : supportedSMs)
    {
        for (int32_t const headDim : supportedHeadDims)
        {
            EXPECT_TRUE(CuteDslFMHAV2Runner::canImplementPaged(
                8, 2, headDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
            EXPECT_TRUE(CuteDslFMHAV2Runner::canImplementPaged(
                8, 2, headDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kSLIDING_CAUSAL));
            EXPECT_TRUE(CuteDslFMHAV2Runner::canImplementPagedRagged(
                8, 2, headDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
            EXPECT_TRUE(CuteDslFMHAV2Runner::canImplementPagedRagged(
                8, 2, headDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kSLIDING_CAUSAL));
            EXPECT_TRUE(CuteDslFMHAV2Runner::canImplement(
                8, 2, headDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
            EXPECT_TRUE(CuteDslFMHAV2Runner::canImplement(
                8, 2, headDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kSLIDING_CAUSAL));
        }

        EXPECT_TRUE(
            CuteDslFMHAV2Runner::canImplement(8, 2, 256, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kPADDING));
        for (int32_t const visionHeadDim : {256, 512})
        {
            EXPECT_TRUE(CuteDslFMHAV2Runner::canImplement(
                8, 2, visionHeadDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kVISION_BLOCK));
            EXPECT_FALSE(CuteDslFMHAV2Runner::canImplementPaged(
                8, 2, visionHeadDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kVISION_BLOCK));
        }
        EXPECT_TRUE(
            CuteDslFMHAV2Runner::canImplement(8, 2, 512, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kPADDING));
        EXPECT_FALSE(CuteDslFMHAV2Runner::canImplementPaged(
            8, 2, 256, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kPADDING));
    }

    EXPECT_FALSE(
        CuteDslFMHAV2Runner::canImplementPaged(8, 2, 64, 102, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
    EXPECT_FALSE(
        CuteDslFMHAV2Runner::canImplementPaged(8, 2, 72, 121, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
    EXPECT_FALSE(
        CuteDslFMHAV2Runner::canImplementPaged(8, 2, 64, 121, DataType::kFLOAT, CuteDslFMHAV2MaskType::kCAUSAL));
    EXPECT_FALSE(
        CuteDslFMHAV2Runner::canImplementPaged(7, 2, 64, 121, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
    EXPECT_FALSE(
        CuteDslFMHAV2Runner::canImplementPagedRagged(8, 2, 64, 121, DataType::kFP8, CuteDslFMHAV2MaskType::kCAUSAL));
    EXPECT_FALSE(
        CuteDslFMHAV2Runner::canImplementPagedRagged(8, 2, 64, 121, DataType::kHALF, CuteDslFMHAV2MaskType::kPADDING));
}

TEST(ContextAttentionTest, pagedAllHeadDimsCausal)
{
    assertPagedCapability(64, CuteDslFMHAV2MaskType::kCAUSAL);
    TestContextAttentionPagedAccuracy(64, 16, 8, 32, 129);
    TestContextAttentionPagedAccuracy(64, 2, 1, 513, 513);

    assertPagedCapability(128, CuteDslFMHAV2MaskType::kCAUSAL);
    TestContextAttentionPagedAccuracy(128, 16, 8, 65, 129);

    assertPagedCapability(256, CuteDslFMHAV2MaskType::kCAUSAL);
    TestContextAttentionPagedAccuracy(256, 8, 2, 65, 191);

    assertPagedCapability(512, CuteDslFMHAV2MaskType::kCAUSAL);
    TestContextAttentionPagedAccuracy(512, 4, 2, 33, 129);
    // D512 GQA breadth and unaligned tails, matching the Gemma4 global-layer ratios.
    TestContextAttentionPagedAccuracy(512, 8, 1, 45, 45);
    TestContextAttentionPagedAccuracy(512, 16, 1, 283, 283);
}

TEST(ContextAttentionTest, pagedD128PageBoundariesAndScale)
{
    for (auto const& [seqLenQ, seqLenK] :
        std::vector<std::pair<int32_t, int32_t>>{{64, 127}, {64, 128}, {65, 129}, {65, 191}, {129, 257}})
    {
        SCOPED_TRACE(::testing::Message() << "Sq=" << seqLenQ << " Sk=" << seqLenK);
        TestContextAttentionPagedAccuracy(128, 16, 8, seqLenQ, seqLenK);
    }
    TestContextAttentionPagedAccuracy(128, 8, 2, 33, 129, INT_MAX, 0.37F);
}

TEST(ContextAttentionTest, pagedD128RaggedScrambledPoisonedPageTables)
{
    // Two batches have independent scrambled K/V pages, partial final pages,
    // different logical Q/K lengths, and an extra -1 page-table entry which
    // must remain unread. Only valid Q rows are compared against the FP32
    // reference because padded rows are outside the kernel contract.
    TestContextAttentionPagedAccuracy(
        128, 8, 2, 65, 257, INT_MAX, std::nullopt, std::vector<int32_t>{33, 65}, std::vector<int32_t>{129, 257});
}

TEST(ContextAttentionTest, pagedNaNPoisonedPaddingDoesNotLeak)
{
    // NaN/Inf in the padded rows, the padded key tail, and the unmapped pages must be masked out
    // rather than propagated through P(0) x V(NaN).
    for (int32_t const headDim : {128, 512})
    {
        SCOPED_TRACE(::testing::Message() << "D=" << headDim);
        TestContextAttentionPagedAccuracy(headDim, 8, 1, 200, 257, INT_MAX, std::nullopt, std::vector<int32_t>{200, 77},
            std::vector<int32_t>{200, 77}, /*nanPoison=*/true);
    }
}

TEST(ContextAttentionTest, pagedD128RaggedSlidingScrambledPoisonedPageTables)
{
    TestContextAttentionPagedAccuracy(
        128, 8, 2, 33, 129, 63, std::nullopt, std::vector<int32_t>{17, 33}, std::vector<int32_t>{65, 129});
}

#if defined(CUTE_DSL_FMHA_BLACKWELL_ENABLED)
class ContextAttentionBlackwellTest : public ::testing::TestWithParam<bool>
{
};

TEST_P(ContextAttentionBlackwellTest, learnedSinks)
{
    // Given ragged paged GQA and per-head FP32 sinks, when causal or sliding prefill runs for every head dimension,
    // then FP16 matches the mathematical reference; FP8 matches its no-sink output times the independent attention
    // mass fraction, and absent sinks match the zero-mass sink path bitwise.
    if (!CuteDslFMHARunner::canImplement(64, getSMVersion()))
    {
        GTEST_SKIP() << "Requires optimized Blackwell SM100/101/103/110.";
    }
    bool const fp8 = GetParam();
    for (int32_t const headDim : {64, 128, 256, 512})
    {
        for (int32_t const window : {INT_MAX, 173})
        {
            for (bool const hasSinks : {false, true})
            {
                SCOPED_TRACE(::testing::Message()
                    << "D=" << headDim << " fp8=" << fp8 << " window=" << window << " sinks=" << hasSinks);
                std::vector<float> const sinks = hasSinks
                    ? std::vector<float>{80.0F, -80.0F, 0.0F, 4.0F, 6.0F, -4.0F, 8.0F, 2.0F}
                    : std::vector<float>{};
                TestContextAttentionPagedAccuracy(headDim, 8, 2, 129, 515, window, 0.37F, {1, 63, 129}, {1, 319, 515},
                    false, true, 0.0F, false, true, fp8, sinks);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Precision, ContextAttentionBlackwellTest, ::testing::Bool(),
    [](testing::TestParamInfo<bool> const& info) { return info.param ? "FP8" : "FP16"; });
#endif

#if defined(CUTE_DSL_FMHA_BLACKWELL_ENABLED)
TEST(ContextAttentionTest, blackwellPackedVisionBlockCapability)
{
    // Given the optimized backend's SM, dtype and head-dimension combinations,
    // When vision-block support is queried,
    // Then only FP16 D512 on SM100/101/103/110 is accepted, and unsupported variants fail preflight.
    for (int32_t const sm : {100, 101, 103, 110, 120, 121})
    {
        for (int32_t const headDim : {64, 128, 256, 512})
        {
            for (DataType const dtype : {DataType::kHALF, DataType::kFP8})
            {
                SCOPED_TRACE(::testing::Message() << "SM=" << sm << " D=" << headDim << " dtype=" << int(dtype));
                bool const supported = sm != 120 && sm != 121 && headDim == 512 && dtype == DataType::kHALF;
                EXPECT_EQ(CuteDslFMHARunner::canImplementPagedRagged(
                              4, 2, headDim, sm, dtype, CuteDslFMHAMaskType::kVISION_BLOCK),
                    supported);
            }
        }
    }
    for (int32_t const headDim : {64, 128, 256, 512})
    {
        for (DataType const dtype : {DataType::kHALF, DataType::kFP8})
        {
            if (headDim != 512 || dtype != DataType::kHALF)
            {
                CuteDslFMHARunner runner(4, 2, headDim, 1, 1, 128);
                EXPECT_FALSE(runner.preflightPagedRagged(nullptr, INT_MAX, dtype, true));
            }
        }
    }
}

TEST(ContextAttentionTest, blackwellPackedVisionBlocks)
{
    // Given unequal packed sequences, cached history, multiple image blocks and tile/page tails,
    // When FP16 D512 causal or sliding vision-block attention runs, including all-text batches,
    // Then output matches padded attention and the exact visible-key mean, without cross-image/sequence leakage.
    if (!CuteDslFMHARunner::canImplement(512, getSMVersion()))
    {
        GTEST_SKIP() << "Requires optimized Blackwell SM100/101/103/110.";
    }
    constexpr int32_t kHEAD_DIM = 512;
    constexpr int32_t kQ_HEADS = 4;
    constexpr int32_t kKV_HEADS = 2;
    constexpr int32_t kTOKENS_PER_PAGE = 128;
    constexpr int32_t kPAGES_PER_SEQ = 3;
    std::vector<int32_t> const qLengths{65, 259, 137, 17};
    std::vector<int32_t> const kvLengths{97, 291, 137, 49};
    int32_t const batchSize = static_cast<int32_t>(qLengths.size());
    int32_t const numPages = batchSize * 2 * kPAGES_PER_SEQ;
    std::vector<int32_t> cuQ{0}, cuKV{0};
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        cuQ.push_back(cuQ.back() + qLengths[batch]);
        cuKV.push_back(cuKV.back() + kvLengths[batch]);
    }
    int32_t const totalQ = cuQ.back();
    int32_t const maxQ = *std::max_element(qLengths.begin(), qLengths.end());
    std::vector<half> const qHost(static_cast<size_t>(totalQ) * kQ_HEADS * kHEAD_DIM, __float2half(0.0F));
    std::vector<half> poolHost(
        static_cast<size_t>(numPages) * kTOKENS_PER_PAGE * kKV_HEADS * kHEAD_DIM, __float2half(112.0F));
    std::vector<int32_t> pageList(static_cast<size_t>(batchSize) * 2 * kPAGES_PER_SEQ);
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        for (int32_t plane = 0; plane < 2; ++plane)
        {
            for (int32_t page = 0; page < kPAGES_PER_SEQ; ++page)
            {
                pageList[(batch * 2 + plane) * kPAGES_PER_SEQ + page]
                    = (batch * 2 + plane) * kPAGES_PER_SEQ + kPAGES_PER_SEQ - page - 1;
            }
        }
        for (int32_t key = 0; key < kvLengths[batch]; ++key)
        {
            for (int32_t head = 0; head < kKV_HEADS; ++head)
            {
                for (int32_t dim = 0; dim < kHEAD_DIM; ++dim)
                {
                    int32_t const kPage = pageList[batch * 2 * kPAGES_PER_SEQ + key / kTOKENS_PER_PAGE];
                    int32_t const vPage = pageList[(batch * 2 + 1) * kPAGES_PER_SEQ + key / kTOKENS_PER_PAGE];
                    poolHost[pagedV2PoolIndex(kPage, key % kTOKENS_PER_PAGE, head, dim, kKV_HEADS, kHEAD_DIM)]
                        = __float2half(0.0F);
                    poolHost[pagedV2PoolIndex(vPage, key % kTOKENS_PER_PAGE, head, dim, kKV_HEADS, kHEAD_DIM)]
                        = __float2half((batch + 1) * 2.0F + key / 512.0F);
                }
            }
        }
    }
    rt::Tensor q({totalQ, kQ_HEADS, kHEAD_DIM}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor pool({numPages, kTOKENS_PER_PAGE, kKV_HEADS, kHEAD_DIM}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor pages({batchSize, 2, kPAGES_PER_SEQ}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor output({totalQ, kQ_HEADS, kHEAD_DIM}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor paddedOutput({totalQ, kQ_HEADS, kHEAD_DIM}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor qPrefix({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor kvPrefix({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor blockBegin({totalQ}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor blockEnd({totalQ}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(q, qHost);
    copyHostToDevice(pool, poolHost);
    copyHostToDevice(pages, pageList);
    copyHostToDevice(qPrefix, cuQ);
    copyHostToDevice(kvPrefix, cuKV);
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    Defer const streamGuard([stream]() { CUDA_CHECK(cudaStreamDestroy(stream)); });
    float const scale = 1.0F / std::sqrt(static_cast<float>(kHEAD_DIM));
    CuteDslFMHARunner runner(kQ_HEADS, kKV_HEADS, kHEAD_DIM, batchSize, maxQ, kPAGES_PER_SEQ * kTOKENS_PER_PAGE);
    for (bool const withImages : {true, false})
    {
        std::vector<int32_t> begins(totalQ, -1), ends(totalQ, -1);
        if (withImages)
        {
            for (int32_t batch = 1; batch < batchSize; ++batch)
            {
                int32_t const history = kvLengths[batch] - qLengths[batch];
                for (auto const& range :
                    {std::pair<int32_t, int32_t>{history + 3, std::min(history + 21, kvLengths[batch] - 1)},
                        std::pair<int32_t, int32_t>{std::max(history + 35, 120), kvLengths[batch] - 1}})
                {
                    for (int32_t position = range.first; position <= range.second; ++position)
                    {
                        begins[cuQ[batch] + position - history] = range.first;
                        ends[cuQ[batch] + position - history] = range.second;
                    }
                }
            }
        }
        copyHostToDevice(blockBegin, begins);
        copyHostToDevice(blockEnd, ends);
        for (int32_t const window : {INT_MAX, 31})
        {
            SCOPED_TRACE(::testing::Message() << "images=" << withImages << " window=" << window);
            ASSERT_TRUE(runner.preflightPagedRagged(stream, window, DataType::kHALF, true)
                && runner.runPagedRagged(q.rawPointer(), pool.rawPointer(), pages.dataPointer<int32_t>(),
                    output.rawPointer(), qPrefix.dataPointer<int32_t>(), kvPrefix.dataPointer<int32_t>(), totalQ, maxQ,
                    numPages, kPAGES_PER_SEQ, kTOKENS_PER_PAGE, stream, scale, window, DataType::kHALF, 1.0F, 1.0F,
                    1.0F, nullptr, blockBegin.dataPointer<int32_t>(), blockEnd.dataPointer<int32_t>()));
            for (int32_t batch = 0; batch < batchSize; ++batch)
            {
                rt::Tensor lengths({2}, rt::DeviceType::kGPU, DataType::kINT32);
                copyHostToDevice(lengths, std::vector<int32_t>{0, kvLengths[batch]});
                CuteDslFMHARunner padded(
                    kQ_HEADS, kKV_HEADS, kHEAD_DIM, 1, qLengths[batch], kPAGES_PER_SEQ * kTOKENS_PER_PAGE);
                size_t const offset = static_cast<size_t>(cuQ[batch]) * kQ_HEADS * kHEAD_DIM;
                ASSERT_TRUE(padded.runPaged(q.dataPointer<half>() + offset, pool.rawPointer(),
                    pages.dataPointer<int32_t>() + batch * 2 * kPAGES_PER_SEQ,
                    paddedOutput.dataPointer<half>() + offset, lengths.dataPointer<int32_t>(), numPages, kPAGES_PER_SEQ,
                    kTOKENS_PER_PAGE, DataType::kHALF, stream, scale, window, false, 1.0F, 1.0F, 1.0F, true, nullptr, 0,
                    nullptr, blockBegin.dataPointer<int32_t>() + cuQ[batch],
                    blockEnd.dataPointer<int32_t>() + cuQ[batch]));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            auto const actual = copyDeviceToHost<half>(output);
            auto const reference = copyDeviceToHost<half>(paddedOutput);
            float maxPaddedError = 0.0F;
            float maxMeanError = 0.0F;
            for (int32_t batch = 0; batch < batchSize; ++batch)
            {
                for (int32_t row = 0; row < qLengths[batch]; ++row)
                {
                    int32_t const index = cuQ[batch] + row;
                    int32_t const position = kvLengths[batch] - qLengths[batch] + row;
                    float sum = 0.0F;
                    int32_t count = 0;
                    // Zero Q/K makes each visible key equiprobable, independently of either GPU mask path.
                    for (int32_t key = 0; key < kvLengths[batch]; ++key)
                    {
                        bool const causal = key <= position && (window == INT_MAX || key >= position - window);
                        bool const sameImage = begins[index] >= 0 && key >= begins[index] && key <= ends[index];
                        if (causal || sameImage)
                        {
                            sum += (batch + 1) * 2.0F + key / 512.0F;
                            ++count;
                        }
                    }
                    for (int32_t col = 0; col < kQ_HEADS * kHEAD_DIM; ++col)
                    {
                        size_t const element = static_cast<size_t>(index) * kQ_HEADS * kHEAD_DIM + col;
                        float const value = __half2float(actual[element]);
                        float const paddedError = std::fabs(value - __half2float(reference[element]));
                        float const meanError = std::fabs(value - sum / count);
                        maxPaddedError = std::isfinite(paddedError) ? std::max(maxPaddedError, paddedError)
                                                                    : std::numeric_limits<float>::infinity();
                        maxMeanError = std::isfinite(meanError) ? std::max(maxMeanError, meanError)
                                                                : std::numeric_limits<float>::infinity();
                    }
                }
            }
            EXPECT_LE(maxPaddedError, 0.01F);
            EXPECT_LE(maxMeanError, 0.01F);
        }
    }
}
#endif

namespace
{

void testRaggedVisionBlocks(int32_t headDim, int32_t window, bool nonzeroQK = false)
{
    int32_t constexpr kSEED = 42;
    int32_t constexpr kHEADS = 4;
    int32_t constexpr kKV_HEADS = 2;
    int32_t constexpr kKV_PLANES = 2;
    int32_t constexpr kPAGE = 128;
    int32_t constexpr kPAGES_PER_SEQ = 3;
    int32_t constexpr kBATCH = 4;
    int32_t constexpr kFLAT_PAGES = kKV_PLANES * kBATCH * kPAGES_PER_SEQ;
#if !defined(CUTE_DSL_FMHA_ENABLED)
    GTEST_SKIP() << "CuTe DSL FMHA is disabled.";
#else
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!CuteDslFMHAV2Runner::canImplementPagedRagged(
            kHEADS, kKV_HEADS, headDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kVISION_BLOCK))
    {
        GTEST_SKIP() << "Ragged vision-block FMHA-v2 is unavailable for D=" << headDim << ", SM=" << smVersion;
    }
#endif
    std::vector<int32_t> const qLengths{145, 67, 41, 97};
    std::vector<int32_t> const kvLengths{273, 195, 41, 163};
    std::vector<int32_t> cuQ{0}, cuKV{0}, packedBegin, packedEnd;
    std::vector<half> packedQ, expectedPadded;
    std::vector<float> expectedReference;
    std::vector<int32_t> pages(kBATCH * kKV_PLANES * kPAGES_PER_SEQ);
    std::vector<half> pool(static_cast<size_t>(kFLAT_PAGES) * kPAGE * kKV_HEADS * headDim,
        __float2half(std::numeric_limits<float>::quiet_NaN()));
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    Defer const streamGuard([stream]() { CUDA_CHECK(cudaStreamDestroy(stream)); });
    for (int32_t batch = 0; batch < kBATCH; ++batch)
    {
        int32_t const length = kvLengths.at(batch);
        int32_t const prefix = length - qLengths.at(batch);
        cuQ.push_back(cuQ.back() + qLengths.at(batch));
        cuKV.push_back(cuKV.back() + length);
        std::vector<int32_t> begin(length, -1), end(length, -2);
        if (batch != 2)
        {
            for (auto const& block : {std::pair{7, 139}, std::pair{151, 270}})
            {
                for (int32_t row = block.first; row <= std::min(block.second, length - 1); ++row)
                {
                    begin.at(row) = block.first;
                    end.at(row) = std::min(block.second, length - 1);
                }
            }
        }
        std::vector<half> q(static_cast<size_t>(length) * kHEADS * headDim, __float2half(0.0F));
        std::vector<half> k(static_cast<size_t>(length) * kKV_HEADS * headDim, __float2half(0.0F));
        std::vector<half> v(k.size());
        if (nonzeroQK)
        {
            std::mt19937 generator(kSEED + batch);
            std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
            auto const randomHalf = [&]() { return __float2half(distribution(generator)); };
            std::generate(q.begin(), q.end(), randomHalf);
            std::generate(k.begin(), k.end(), randomHalf);
            std::generate(v.begin(), v.end(), randomHalf);
        }
        else
        {
            for (int32_t row = 0; row < length; ++row)
            {
                for (int32_t head = 0; head < kKV_HEADS; ++head)
                {
                    for (int32_t dim = 0; dim < headDim; ++dim)
                    {
                        size_t const index = (static_cast<size_t>(row) * kKV_HEADS + head) * headDim + dim;
                        v.at(index) = __float2half(
                            batch * 0.5F + head * 0.125F + static_cast<float>((row * (dim % 7 + 1)) % 281) / 281.0F);
                    }
                }
            }
        }
        for (int32_t plane = 0; plane < kKV_PLANES; ++plane)
        {
            for (int32_t page = 0; page < kPAGES_PER_SEQ; ++page)
            {
                int32_t const physical = plane * kBATCH * kPAGES_PER_SEQ + (kBATCH - 1 - batch) * kPAGES_PER_SEQ
                    + (kPAGES_PER_SEQ - 1 - page);
                pages.at((batch * kKV_PLANES + plane) * kPAGES_PER_SEQ + page) = physical;
                for (int32_t row = page * kPAGE; row < std::min((page + 1) * kPAGE, length); ++row)
                {
                    auto const& source = plane == 0 ? k : v;
                    size_t const sourceOffset = static_cast<size_t>(row) * kKV_HEADS * headDim;
                    size_t const targetOffset = pagedV2PoolIndex(physical, row % kPAGE, 0, 0, kKV_HEADS, headDim);
                    std::copy_n(source.begin() + sourceOffset, kKV_HEADS * headDim, pool.begin() + targetOffset);
                }
            }
        }
        size_t const start = static_cast<size_t>(prefix) * kHEADS * headDim;
        packedQ.insert(packedQ.end(), q.begin() + start, q.end());
        if (!nonzeroQK)
        {
            rt::Tensor qDense({1, length, kHEADS, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
            rt::Tensor kDense({1, length, kKV_HEADS, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
            rt::Tensor vDense({1, length, kKV_HEADS, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
            rt::Tensor output({1, length, kHEADS, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
            rt::Tensor cuDense({2}, rt::DeviceType::kGPU, DataType::kINT32);
            rt::Tensor beginDense({1, length}, rt::DeviceType::kGPU, DataType::kINT32);
            rt::Tensor endDense({1, length}, rt::DeviceType::kGPU, DataType::kINT32);
            copyHostToDevice(qDense, q);
            copyHostToDevice(kDense, k);
            copyHostToDevice(vDense, v);
            copyHostToDevice(cuDense, std::vector<int32_t>{0, length});
            copyHostToDevice(beginDense, begin);
            copyHostToDevice(endDense, end);
            CuteDslFMHAV2Runner padded(kHEADS, kKV_HEADS, headDim, 1, length, length);
            ASSERT_TRUE(padded.runVisionBlock(qDense.rawPointer(), kDense.rawPointer(), vDense.rawPointer(),
                output.rawPointer(), cuDense.dataPointer<int32_t>(), beginDense.dataPointer<int32_t>(),
                endDense.dataPointer<int32_t>(), stream, 1.0F / std::sqrt(static_cast<float>(headDim)),
                window == INT_MAX ? -1 : window));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto const paddedResult = copyDeviceToHost<half>(output);
            expectedPadded.insert(expectedPadded.end(), paddedResult.begin() + start, paddedResult.end());
        }
        packedBegin.insert(packedBegin.end(), begin.begin() + prefix, begin.end());
        packedEnd.insert(packedEnd.end(), end.begin() + prefix, end.end());
        float const attentionScale = 1.0F / std::sqrt(static_cast<float>(headDim));
        std::vector<float> weights(length);
        for (int32_t row = prefix; row < length; ++row)
        {
            for (int32_t head = 0; head < kHEADS; ++head)
            {
                float maximum = -std::numeric_limits<float>::infinity();
                for (int32_t key = 0; key < length; ++key)
                {
                    bool const causal = key <= row && (window == INT_MAX || row - key <= window);
                    bool const sameImage = begin.at(row) >= 0 && key >= begin.at(row) && key <= end.at(row);
                    float score = -std::numeric_limits<float>::infinity();
                    if (causal || sameImage)
                    {
                        score = 0.0F;
                        if (nonzeroQK)
                        {
                            for (int32_t dim = 0; dim < headDim; ++dim)
                            {
                                size_t const qIndex = (static_cast<size_t>(row) * kHEADS + head) * headDim + dim;
                                size_t const kIndex
                                    = (static_cast<size_t>(key) * kKV_HEADS + head / (kHEADS / kKV_HEADS)) * headDim
                                    + dim;
                                score += __half2float(q.at(qIndex)) * __half2float(k.at(kIndex));
                            }
                            score *= attentionScale;
                        }
                    }
                    weights.at(key) = score;
                    maximum = std::max(maximum, score);
                }
                float denominator = 0.0F;
                for (float& weight : weights)
                {
                    weight = std::exp(weight - maximum);
                    denominator += weight;
                }
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    float sum = 0.0F;
                    for (int32_t key = 0; key < length; ++key)
                    {
                        size_t const vIndex
                            = (static_cast<size_t>(key) * kKV_HEADS + head / (kHEADS / kKV_HEADS)) * headDim + dim;
                        sum += weights.at(key) * __half2float(v.at(vIndex));
                    }
                    expectedReference.push_back(sum / denominator);
                }
            }
        }
    }
    int32_t const total = cuQ.back();
    rt::Tensor q({total, kHEADS, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor output({total, kHEADS, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kv({kFLAT_PAGES, kPAGE, kKV_HEADS, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor pageTable({kBATCH, kKV_PLANES, kPAGES_PER_SEQ}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor qOffsets({kBATCH + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor kvOffsets({kBATCH + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor begin({total}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor end({total}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(q, packedQ);
    copyHostToDevice(kv, pool);
    copyHostToDevice(pageTable, pages);
    copyHostToDevice(qOffsets, cuQ);
    copyHostToDevice(kvOffsets, cuKV);
    copyHostToDevice(begin, packedBegin);
    copyHostToDevice(end, packedEnd);
    CuteDslFMHAV2Runner runner(kHEADS, kKV_HEADS, headDim, kBATCH, 145, kPAGES_PER_SEQ * kPAGE);
    ASSERT_TRUE(runner.preflightPagedRagged(stream, window, true));
    CuteDslFMHAV2Runner::PagedRaggedParams params{};
    params.qPtr = q.rawPointer();
    params.pagedKVPoolPtr = kv.rawPointer();
    params.kvCachePageList = pageTable.dataPointer<int32_t>();
    params.oPtr = output.rawPointer();
    params.cuQSeqLens = qOffsets.dataPointer<int32_t>();
    params.cuKVSeqLens = kvOffsets.dataPointer<int32_t>();
    params.totalQSeqLen = total;
    params.maxQSeqLen = 145;
    params.numFlatPages = kFLAT_PAGES;
    params.maxPagesPerSeq = kPAGES_PER_SEQ;
    params.tokensPerPage = kPAGE;
    params.stream = stream;
    params.attentionScale = 1.0F / std::sqrt(static_cast<float>(headDim));
    params.slidingWindowSize = window;
    params.bidirectionalBlockBegin = begin.dataPointer<int32_t>();
    params.bidirectionalBlockEnd = end.dataPointer<int32_t>();
    ASSERT_TRUE(runner.runPagedRagged(params));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto const actual = copyDeviceToHost<half>(output);
    float maxAbsError = 0.0F;
    for (size_t i = 0; i < actual.size(); ++i)
    {
        if (!nonzeroQK)
        {
            ASSERT_NEAR(__half2float(actual.at(i)), __half2float(expectedPadded.at(i)), 3e-3F) << "index=" << i;
        }
        maxAbsError = std::max(maxAbsError, std::abs(__half2float(actual.at(i)) - expectedReference.at(i)));
        ASSERT_NEAR(__half2float(actual.at(i)), expectedReference.at(i), 3e-3F) << "index=" << i;
    }
    if (nonzeroQK)
    {
        std::cout << "Ragged vision nonuniform softmax: D=" << headDim << " window=" << window
                  << " max_abs_error=" << maxAbsError << std::endl;
    }
}

} // namespace

TEST(ContextAttentionTest, pagedPackedRaggedVisionCapabilities)
{
    // Given vision attention configurations, when support is queried, then only shipped FP16 D256/D512
    // variants with supported SMs and integral GQA groups are accepted.
    for (int32_t const sm : {75, 80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121})
    {
        for (int32_t const headDim : {64, 128, 256, 512})
        {
            for (auto const dtype : {DataType::kHALF, DataType::kBF16, DataType::kFP8})
            {
                for (int32_t const qHeads : {0, 3, 8})
                {
                    bool supported = false;
#if defined(CUTE_DSL_FMHA_ENABLED)
                    supported
                        = sm != 75 && (headDim == 256 || headDim == 512) && dtype == DataType::kHALF && qHeads == 8;
#endif
                    EXPECT_EQ(CuteDslFMHAV2Runner::canImplementPagedRagged(
                                  qHeads, 2, headDim, sm, dtype, CuteDslFMHAV2MaskType::kVISION_BLOCK),
                        supported)
                        << "SM=" << sm << " D=" << headDim << " Hq=" << qHeads;
                }
            }
        }
    }
}

TEST(ContextAttentionTest, pagedPackedRaggedVisionBlocks)
{
    // Given heterogeneous text/image sequences with cached prefixes and multiple images crossing tiles/pages,
    // when packed paged vision attention runs, then each output matches padded attention and the exact mask mean.
    for (int32_t const headDim : {256, 512})
    {
        for (int32_t const window : {15, INT_MAX})
        {
            testRaggedVisionBlocks(headDim, window);
        }
    }
}

TEST(ContextAttentionTest, pagedPackedRaggedVisionNonuniformSoftmax)
{
    // Given deterministic nonzero Q/K/V with heterogeneous cached prefixes and vision blocks crossing tiles/pages,
    // when packed paged attention runs, then its nonuniform softmax output matches an independent FP32 reference.
    for (int32_t const headDim : {256, 512})
    {
        for (int32_t const window : {15, INT_MAX})
        {
            SCOPED_TRACE(::testing::Message() << "D=" << headDim << " window=" << window);
            testRaggedVisionBlocks(headDim, window, true);
        }
    }
}

TEST(ContextAttentionTest, pagedPackedRaggedCold)
{
    TestContextAttentionPagedAccuracy(
        64, 8, 2, 128, 128, INT_MAX, std::nullopt, std::vector<int32_t>{128}, std::vector<int32_t>{128}, false, true);
}

TEST(ContextAttentionTest, pagedPackedRaggedChunk)
{
    TestContextAttentionPagedAccuracy(
        64, 8, 2, 65, 257, INT_MAX, std::nullopt, std::vector<int32_t>{65}, std::vector<int32_t>{257}, false, true);
}

TEST(ContextAttentionTest, pagedPackedRaggedHeterogeneous)
{
    TestContextAttentionPagedAccuracy(64, 8, 2, 129, 129, INT_MAX, std::nullopt, std::vector<int32_t>{1, 63, 129},
        std::vector<int32_t>{1, 63, 129}, false, true);
}

TEST(ContextAttentionTest, pagedPackedRaggedSlidingWindow)
{
    TestContextAttentionPagedAccuracy(
        64, 8, 2, 129, 129, 63, std::nullopt, std::vector<int32_t>{129}, std::vector<int32_t>{129}, false, true);
}

TEST(ContextAttentionTest, pagedPackedRaggedD128)
{
    TestContextAttentionPagedAccuracy(
        128, 8, 2, 65, 129, INT_MAX, std::nullopt, std::vector<int32_t>{65}, std::vector<int32_t>{65}, false, true);
}

TEST(ContextAttentionTest, pagedPackedRaggedD256)
{
    TestContextAttentionPagedAccuracy(
        256, 8, 2, 33, 129, INT_MAX, std::nullopt, std::vector<int32_t>{33}, std::vector<int32_t>{33}, false, true);
}

TEST(ContextAttentionTest, pagedPackedRaggedD512)
{
    TestContextAttentionPagedAccuracy(
        512, 4, 1, 17, 129, INT_MAX, std::nullopt, std::vector<int32_t>{17}, std::vector<int32_t>{17}, false, true);
}

TEST(ContextAttentionTest, pagedPackedRaggedAttentionSinks)
{
    // Given packed ragged FP16 Q with heterogeneous lengths, GQA paged KV and per-head sink logits,
    // When causal and sliding kernels process multi-page tails,
    // Then every output matches FP32 softmax with one sink denominator term per query/head and no sink value.
    std::vector<float> const attentionSinks{-20.0F, -3.0F, -0.125F, 0.0F, 0.25F, 3.0F, 20.0F, 1.5F};
    for (int32_t const headDim : {64, 128, 256, 512})
    {
        TestContextAttentionPagedAccuracy(headDim, 8, 2, 129, 257, INT_MAX, std::nullopt, {1, 65, 129}, {129, 193, 257},
            false, true, 0.0F, false, false, false, attentionSinks);
        TestContextAttentionPagedAccuracy(headDim, 8, 2, 129, 257, 63, std::nullopt, {1, 65, 129}, {129, 193, 257},
            false, true, 0.0F, false, false, false, attentionSinks);
    }
}

// Skip-softmax paged FMHA-v2: every supported head dim, GQA, ragged lengths (per-sequence lambda = S / seqlen_kv)
// and a chunked-prefill offset. Sequences span several KV tiles so tiles after the diagonal actually skip. The
// non-hot GQA heads see random scores, where a skipped tile may legitimately carry up to Bc * lambda of the row's
// max weight, so lambda stays small enough to keep that inside the 0.1 gate.
TEST(ContextAttentionTest, pagedAllHeadDimsSkipSoftmax)
{
    float constexpr kLambda = 0.001F;
    TestContextAttentionPagedAccuracy(64, 8, 4, 768, 768, INT_MAX, std::nullopt, {}, {}, false, false, kLambda);
    TestContextAttentionPagedAccuracy(128, 8, 2, 768, 768, INT_MAX, std::nullopt, {}, {}, false, false, kLambda);
    TestContextAttentionPagedAccuracy(128, 8, 2, 300, 1100, INT_MAX, std::nullopt, std::vector<int32_t>{300, 200},
        std::vector<int32_t>{1100, 700}, false, false, kLambda);
    TestContextAttentionPagedAccuracy(256, 8, 2, 768, 768, INT_MAX, std::nullopt, {}, {}, false, false, kLambda);
    TestContextAttentionPagedAccuracy(512, 4, 1, 768, 768, INT_MAX, std::nullopt, {}, {}, false, false, kLambda);
}

// A tiny S never trips the predicate: with the table bound the kernel must reproduce the dense result on
// unstructured data (dense tolerance), covering the vote/no-skip path.
TEST(ContextAttentionTest, pagedSkipSoftmaxNeverFiresMatchesDense)
{
    for (int32_t const headDim : {64, 128, 256, 512})
    {
        SCOPED_TRACE(::testing::Message() << "D=" << headDim);
        TestContextAttentionPagedAccuracy(headDim, 8, 2, 65, 257, INT_MAX, std::nullopt, {}, {}, false, false, 1e-6F,
            /*skipSoftmaxExpectDense=*/true);
    }
}

// S >= seqlen_kv (lambda >= 1 would pass every tile): the in-kernel guard forces the threshold to -inf, so the
// kernel reproduces the dense result although the table is bound. Covers S == L and S > L — the same guard that
// runs unbound batches dense via the {0, 0} row.
TEST(ContextAttentionTest, pagedSkipSoftmaxGuardMatchesDense)
{
    for (int32_t const headDim : {64, 128, 256, 512})
    {
        SCOPED_TRACE(::testing::Message() << "D=" << headDim);
        TestContextAttentionPagedAccuracy(
            headDim, 8, 2, 65, 257, INT_MAX, std::nullopt, {}, {}, false, false, 1.0F, /*skipSoftmaxExpectDense=*/true);
    }
    TestContextAttentionPagedAccuracy(
        128, 8, 2, 65, 257, INT_MAX, std::nullopt, {}, {}, false, false, 2.0F, /*skipSoftmaxExpectDense=*/true);
}

TEST(ContextAttentionTest, pagedAllHeadDimsSliding)
{
    assertPagedCapability(64, CuteDslFMHAV2MaskType::kSLIDING_CAUSAL);
    TestContextAttentionPagedAccuracy(64, 16, 8, 64, 191, 31);

    assertPagedCapability(128, CuteDslFMHAV2MaskType::kSLIDING_CAUSAL);
    TestContextAttentionPagedAccuracy(128, 8, 2, 65, 191, 63);

    assertPagedCapability(256, CuteDslFMHAV2MaskType::kSLIDING_CAUSAL);
    TestContextAttentionPagedAccuracy(256, 8, 2, 33, 129, 31);
    // A window wider than Br + Bc leaves complete KV tiles valid for every row
    // in the Q tile, exercising the paged sliding kernel's unmasked interior.
    TestContextAttentionPagedAccuracy(256, 8, 2, 33, 129, 128);

    assertPagedCapability(512, CuteDslFMHAV2MaskType::kSLIDING_CAUSAL);
    TestContextAttentionPagedAccuracy(512, 4, 2, 17, 129, 31);
}

// Test cases with different head ratios (similar to XQA tests)

TEST(ContextAttentionTest, accuracyKVRatio1_Causal)
{
    // MHA: num_Q_heads == num_KV_heads
    TestContextAttentionAccuracy(1, 512, 8, 8, 128, true);
    TestContextAttentionAccuracy(2, 256, 16, 16, 64, true);
    TestContextAttentionAccuracy(4, 512, 4, 4, 128, true);
    TestContextAttentionAccuracy(1, 512, 4, 4, 256, true);
    TestContextAttentionAccuracy(1, 64, 4, 4, 256, true);
}

TEST(ContextAttentionTest, accuracyKVRatio3_Causal)
{
    // GQA with ratio 3
    TestContextAttentionAccuracy(1, 512, 24, 8, 64, true);
    TestContextAttentionAccuracy(4, 512, 12, 4, 128, true);
}

TEST(ContextAttentionTest, accuracyKVRatio4_Causal)
{
    // GQA with ratio 4
    TestContextAttentionAccuracy(1, 132, 32, 8, 64, true);
    TestContextAttentionAccuracy(2, 260, 32, 8, 128, true);
    TestContextAttentionAccuracy(4, 520, 16, 4, 128, true);
    TestContextAttentionAccuracy(1, 512, 16, 4, 256, true);
    TestContextAttentionAccuracy(2, 48, 16, 4, 256, true);
}

TEST(ContextAttentionTest, accuracyKVRatio7_Causal)
{
    // GQA with ratio 7
    TestContextAttentionAccuracy(1, 784, 28, 4, 64, true);
    TestContextAttentionAccuracy(2, 512, 14, 2, 128, true);
    TestContextAttentionAccuracy(4, 256, 14, 2, 128, true);
}

TEST(ContextAttentionTest, accuracyKVRatio8_Causal)
{
    // GQA with ratio 8
    TestContextAttentionAccuracy(1, 128, 32, 4, 64, true);
    TestContextAttentionAccuracy(2, 256, 16, 2, 128, true);
    TestContextAttentionAccuracy(4, 512, 16, 2, 128, true);
    TestContextAttentionAccuracy(2, 256, 16, 2, 256, true);
}

TEST(ContextAttentionTest, paddedLayout_VariableSequenceLengths)
{
    TestContextAttentionAccuracy({0, 64, 160}, 8, 2, 128, 128);
}

// Long sequence tests
TEST(ContextAttentionTest, longSequence_Causal)
{
    TestContextAttentionAccuracy(1, 1024, 12, 4, 128, true);
    TestContextAttentionAccuracy(1, 1024, 12, 2, 128, true);
    TestContextAttentionAccuracy(1, 2048, 24, 3, 64, true);
    TestContextAttentionAccuracy(1, 1024, 8, 2, 256, true);
}

// Convenience wrapper for compact layout (variable sequence lengths, non-causal)
void TestContextAttentionCompactAccuracy(std::vector<int32_t> const& cuSeqlens, int32_t numQHeads, int32_t numKVHeads,
    int32_t headSize, int32_t maxSeqLen, std::optional<float> attentionScale = std::nullopt)
{
    TestContextAttentionAccuracy(cuSeqlens, numQHeads, numKVHeads, headSize, maxSeqLen, true, false, attentionScale);
}

TEST(ContextAttentionTest, compactLayout_NonCausal)
{
    // VIT attention with compact layout and variable sequence lengths (non-causal)
    TestContextAttentionCompactAccuracy({0, 32, 60, 88, 128}, 16, 16, 64, 128);
    TestContextAttentionCompactAccuracy({0, 16, 64}, 16, 16, 72, 128);
    TestContextAttentionCompactAccuracy({0, 100, 200, 300}, 8, 8, 80, 512);
}

TEST(ContextAttentionTest, configurableScale)
{
    float constexpr kIDENTITY_SCALE = 1.0F;
    float constexpr kCUSTOM_SCALE = 0.37F;

    TestContextAttentionAccuracy(1, 64, 8, 8, 64, true, kIDENTITY_SCALE);
    TestContextAttentionAccuracy(1, 64, 8, 2, 128, true, kCUSTOM_SCALE);
    TestContextAttentionAccuracy(1, 256, 8, 1, 256, true, kIDENTITY_SCALE);
    TestContextAttentionAccuracy(1, 256, 8, 1, 256, true, kCUSTOM_SCALE);
    TestContextAttentionCompactAccuracy({0, 16, 48}, 8, 8, 64, 48, kIDENTITY_SCALE);
    TestContextAttentionCompactAccuracy({0, 24, 64}, 8, 8, 80, 64, kCUSTOM_SCALE);
}

namespace
{

void fillNonCausalVarlenReference(std::vector<half> const& qInput, std::vector<half> const& kInput,
    std::vector<half> const& vInput, std::vector<half>& output, std::vector<int32_t> const& qLens,
    std::vector<int32_t> const& kvLens, int32_t seqLenQ, int32_t seqLenK, int32_t numQHeads, int32_t numKVHeads,
    int32_t headSize)
{
    int32_t const batchSize = static_cast<int32_t>(qLens.size());
    float const softmaxScale = 1.0F / std::sqrt(static_cast<float>(headSize));
    std::fill(output.begin(), output.end(), __float2half(0.0F));

    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t qPos = 0; qPos < qLens[static_cast<size_t>(b)]; ++qPos)
        {
            for (int32_t qHead = 0; qHead < numQHeads; ++qHead)
            {
                int32_t const kvHead = qHead * numKVHeads / numQHeads;
                float rowMax = -std::numeric_limits<float>::infinity();
                for (int32_t kPos = 0; kPos < kvLens[static_cast<size_t>(b)]; ++kPos)
                {
                    float score = 0.0F;
                    for (int32_t d = 0; d < headSize; ++d)
                    {
                        size_t const qIdx
                            = (((static_cast<size_t>(b) * seqLenQ + qPos) * numQHeads + qHead) * headSize) + d;
                        size_t const kIdx
                            = (((static_cast<size_t>(b) * seqLenK + kPos) * numKVHeads + kvHead) * headSize) + d;
                        score += __half2float(qInput[qIdx]) * __half2float(kInput[kIdx]);
                    }
                    rowMax = std::max(rowMax, score * softmaxScale);
                }

                float rowSum = 0.0F;
                std::vector<float> weights(static_cast<size_t>(kvLens[static_cast<size_t>(b)]));
                for (int32_t kPos = 0; kPos < kvLens[static_cast<size_t>(b)]; ++kPos)
                {
                    float score = 0.0F;
                    for (int32_t d = 0; d < headSize; ++d)
                    {
                        size_t const qIdx
                            = (((static_cast<size_t>(b) * seqLenQ + qPos) * numQHeads + qHead) * headSize) + d;
                        size_t const kIdx
                            = (((static_cast<size_t>(b) * seqLenK + kPos) * numKVHeads + kvHead) * headSize) + d;
                        score += __half2float(qInput[qIdx]) * __half2float(kInput[kIdx]);
                    }
                    float const weight = std::exp((score * softmaxScale) - rowMax);
                    weights[static_cast<size_t>(kPos)] = weight;
                    rowSum += weight;
                }

                for (int32_t d = 0; d < headSize; ++d)
                {
                    float acc = 0.0F;
                    for (int32_t kPos = 0; kPos < kvLens[static_cast<size_t>(b)]; ++kPos)
                    {
                        size_t const vIdx
                            = (((static_cast<size_t>(b) * seqLenK + kPos) * numKVHeads + kvHead) * headSize) + d;
                        acc += (weights[static_cast<size_t>(kPos)] / rowSum) * __half2float(vInput[vIdx]);
                    }
                    size_t const outIdx
                        = (((static_cast<size_t>(b) * seqLenQ + qPos) * numQHeads + qHead) * headSize) + d;
                    output[outIdx] = __float2half(acc);
                }
            }
        }
    }
}

void TestContextAttentionDenoisePaddingVarlen(
    std::vector<int32_t> const& qLens, std::vector<int32_t> const& kvLens, int32_t headSize = 256)
{
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    int32_t constexpr numQHeads = 16;
    int32_t constexpr numKVHeads = 8;
    int32_t const batchSize = static_cast<int32_t>(qLens.size());
    int32_t const seqLenQ = *std::max_element(qLens.begin(), qLens.end());
    int32_t const seqLenK = *std::max_element(kvLens.begin(), kvLens.end());

    if (!CuteDslFMHAV2Runner::canImplement(
            numQHeads, numKVHeads, headSize, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kPADDING))
    {
        GTEST_SKIP() << "FMHA-v2 CuTe DSL PADDING is not supported for headSize=" << headSize << ", SM=" << smVersion;
    }

    size_t const qSize = static_cast<size_t>(batchSize) * seqLenQ * numQHeads * headSize;
    size_t const kvSize = static_cast<size_t>(batchSize) * seqLenK * numKVHeads * headSize;
    size_t const outSize = qSize;

    std::vector<half> qInput(qSize);
    std::vector<half> kInput(kvSize);
    std::vector<half> vInput(kvSize);
    uniformFloatInitialization(qInput, -1.0F, 1.0F);
    uniformFloatInitialization(kInput, -1.0F, 1.0F);
    uniformFloatInitialization(vInput, -1.0F, 1.0F);

    std::vector<int32_t> cuQ(static_cast<size_t>(batchSize + 1), 0);
    std::vector<int32_t> cuKV(static_cast<size_t>(batchSize + 1), 0);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        cuQ[static_cast<size_t>(b + 1)] = cuQ[static_cast<size_t>(b)] + qLens[static_cast<size_t>(b)];
        cuKV[static_cast<size_t>(b + 1)] = cuKV[static_cast<size_t>(b)] + kvLens[static_cast<size_t>(b)];
    }

    rt::Tensor qTensor({batchSize, seqLenQ, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kTensor({batchSize, seqLenK, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({batchSize, seqLenK, numKVHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor oTensor({batchSize, seqLenQ, numQHeads, headSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuQTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    CUDA_CHECK(cudaMemcpy(qTensor.rawPointer(), qInput.data(), qSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kTensor.rawPointer(), kInput.data(), kvSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vTensor.rawPointer(), vInput.data(), kvSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuQTensor.rawPointer(), cuQ.data(), static_cast<size_t>(batchSize + 1) * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuKVTensor.rawPointer(), cuKV.data(), static_cast<size_t>(batchSize + 1) * sizeof(int32_t),
        cudaMemcpyHostToDevice));

    rt::Tensor blockBeginTensor({batchSize, seqLenQ}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor blockEndTensor({batchSize, seqLenQ}, rt::DeviceType::kGPU, DataType::kINT32);
    kernel::launchBuildPaddingBlockRanges(cuKVTensor.dataPointer<int32_t>(), blockBeginTensor.dataPointer<int32_t>(),
        blockEndTensor.dataPointer<int32_t>(), batchSize, seqLenQ, nullptr);

    CuteDslFMHAV2Runner runner(numQHeads, numKVHeads, headSize, batchSize, seqLenQ, seqLenK);
    ASSERT_TRUE(runner.runPadding(qTensor.rawPointer(), kTensor.rawPointer(), vTensor.rawPointer(),
        oTensor.rawPointer(), cuQTensor.dataPointer<int32_t>(), cuKVTensor.dataPointer<int32_t>(), nullptr,
        1.0F / std::sqrt(static_cast<float>(headSize)), blockBeginTensor.dataPointer<int32_t>(),
        blockEndTensor.dataPointer<int32_t>()));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::vector<half> expected(outSize);
    fillNonCausalVarlenReference(
        qInput, kInput, vInput, expected, qLens, kvLens, seqLenQ, seqLenK, numQHeads, numKVHeads, headSize);
    std::vector<half> actual(outSize);
    CUDA_CHECK(cudaMemcpy(actual.data(), oTensor.rawPointer(), outSize * sizeof(half), cudaMemcpyDeviceToHost));

    for (int32_t b = 0; b < batchSize; ++b)
    {
        for (int32_t qPos = 0; qPos < qLens[static_cast<size_t>(b)]; ++qPos)
        {
            for (int32_t h = 0; h < numQHeads; ++h)
            {
                for (int32_t d = 0; d < headSize; ++d)
                {
                    size_t const idx = (((static_cast<size_t>(b) * seqLenQ + qPos) * numQHeads + h) * headSize) + d;
                    ASSERT_TRUE(isclose(actual[idx], expected[idx], 1e-2F, 1e-2F))
                        << "DiffusionGemma denoise PADDING mismatch at b=" << b << " q=" << qPos << " h=" << h
                        << " d=" << d << " expected=" << __half2float(expected[idx])
                        << " actual=" << __half2float(actual[idx]);
                }
            }
        }
    }
}

} // namespace

TEST(ContextAttentionTest, diffusionGemmaDenoisePaddingVarlen)
{
    TestContextAttentionDenoisePaddingVarlen({4}, {24});
    TestContextAttentionDenoisePaddingVarlen({4, 3}, {24, 11});
    TestContextAttentionDenoisePaddingVarlen({4}, {24}, 512);
    TestContextAttentionDenoisePaddingVarlen({4, 3}, {24, 11}, 512);
    TestContextAttentionDenoisePaddingVarlen({70, 33}, {135, 33}, 512);
}
