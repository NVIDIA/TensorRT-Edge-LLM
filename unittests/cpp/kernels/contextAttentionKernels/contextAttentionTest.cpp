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
#include <limits>
#include <optional>
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

using namespace nvinfer1;
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
    int32_t headDim, float attentionScale, int32_t windowSizeLeft)
{
    std::vector<float> output(static_cast<size_t>(batchSize) * seqLenQ * numQHeads * headDim);
    int32_t const groupSize = numQHeads / numKVHeads;
    std::vector<float> weights(static_cast<size_t>(seqLenKCapacity));

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
                float rowMax = -std::numeric_limits<float>::infinity();
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
                    rowMax = std::max(rowMax, score * attentionScale);
                }

                float rowSum = 0.0F;
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
                    float const weight = std::exp(score * attentionScale - rowMax);
                    weights[static_cast<size_t>(kPos)] = weight;
                    rowSum += weight;
                }

                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    float value = 0.0F;
                    for (int32_t kPos = keyBegin; kPos <= keyEnd; ++kPos)
                    {
                        size_t const vIdx
                            = pagedV2BshdIndex(batch, kPos, kvHead, dim, seqLenKCapacity, numKVHeads, headDim);
                        value += weights[static_cast<size_t>(kPos)] * __half2float(v[vIdx]) / rowSum;
                    }
                    output[pagedV2BshdIndex(batch, qPos, qHead, dim, seqLenQ, numQHeads, headDim)] = value;
                }
            }
        }
    }
    return output;
}

void TestContextAttentionPagedAccuracy(int32_t headDim, int32_t numQHeads, int32_t numKVHeads, int32_t seqLenQ,
    int32_t seqLenKCapacity, int32_t windowSizeLeft = INT_MAX,
    std::optional<float> requestedAttentionScale = std::nullopt, std::vector<int32_t> seqLensQ = {},
    std::vector<int32_t> seqLensK = {}, bool nanPoison = false)
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

    rt::Tensor qTensor({batchSize, seqLenQ, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor poolTensor({numFlatPages, kTOKENS_PER_PAGE, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor pageListTensor({batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor outputPaged({batchSize, seqLenQ, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuQSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    std::vector<int32_t> cuQSeqLens(static_cast<size_t>(batchSize + 1), 0);
    std::vector<int32_t> cuKVSeqLens(static_cast<size_t>(batchSize + 1), 0);
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        cuQSeqLens[static_cast<size_t>(batch + 1)]
            = cuQSeqLens[static_cast<size_t>(batch)] + seqLensQ[static_cast<size_t>(batch)];
        cuKVSeqLens[static_cast<size_t>(batch + 1)]
            = cuKVSeqLens[static_cast<size_t>(batch)] + seqLensK[static_cast<size_t>(batch)];
    }
    copyHostToDevice(qTensor, qHost);
    copyHostToDevice(poolTensor, poolHost);
    copyHostToDevice(pageListTensor, pageList);
    copyHostToDevice(cuQSeqLensTensor, cuQSeqLens);
    copyHostToDevice(cuKVSeqLensTensor, cuKVSeqLens);

    CuteDslFMHAV2Runner runner(numQHeads, numKVHeads, headDim, batchSize, seqLenQ, capacity);
    ASSERT_TRUE(runner.runPaged(qTensor.rawPointer(), poolTensor.rawPointer(), pageListTensor.dataPointer<int32_t>(),
        outputPaged.rawPointer(), cuQSeqLensTensor.dataPointer<int32_t>(), cuKVSeqLensTensor.dataPointer<int32_t>(),
        numFlatPages, maxPagesPerSeq, kTOKENS_PER_PAGE, nullptr, attentionScale, windowSizeLeft));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaGetLastError());

    auto const actual = copyDeviceToHost<half>(outputPaged);
    auto const expected = computePagedV2Reference(qHost, kHost, vHost, batchSize, seqLenQ, seqLenKCapacity, seqLensQ,
        seqLensK, numQHeads, numKVHeads, headDim, attentionScale, windowSizeLeft);
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
                    float const actualValue = __half2float(actual[idx]);
                    ASSERT_TRUE(isclose(actualValue, expected[idx], 1e-2F, 1e-2F))
                        << "Paged FMHA-v2 mismatch at index=" << idx << " batch=" << batch << " seq=" << seq
                        << " D=" << headDim << " SqCapacity=" << seqLenQ
                        << " actualSq=" << seqLensQ[static_cast<size_t>(batch)] << " SkCapacity=" << seqLenKCapacity
                        << " actualSk=" << seqLensK[static_cast<size_t>(batch)] << " windowLeft=" << windowSizeLeft
                        << " expected=" << expected[idx] << " actual=" << actualValue;
                    closeWithin1e3 += isclose(actualValue, expected[idx], 1e-3F, 1e-3F);
                    ASSERT_FALSE(std::isnan(actualValue));
                    ASSERT_FALSE(std::isinf(actualValue));
                    ++validElements;
                }
            }
        }
    }
    EXPECT_GT(static_cast<float>(closeWithin1e3) / static_cast<float>(validElements), 0.9F);
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
    std::vector<int32_t> const supportedSMs{80, 86, 87, 89, 90, 100, 101, 110, 120, 121};
    std::vector<int32_t> const supportedHeadDims{64, 128, 256, 512};

    for (int32_t const smVersion : supportedSMs)
    {
        for (int32_t const headDim : supportedHeadDims)
        {
            EXPECT_TRUE(CuteDslFMHAV2Runner::canImplementPaged(
                8, 2, headDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
            EXPECT_TRUE(CuteDslFMHAV2Runner::canImplementPaged(
                8, 2, headDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kSLIDING_CAUSAL));
            EXPECT_TRUE(CuteDslFMHAV2Runner::canImplement(
                8, 2, headDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
            EXPECT_TRUE(CuteDslFMHAV2Runner::canImplement(
                8, 2, headDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kSLIDING_CAUSAL));
        }

        EXPECT_TRUE(
            CuteDslFMHAV2Runner::canImplement(8, 2, 256, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kPADDING));
        // The vision-block overlay is a dense contract at both shipped head dims; the paged
        // kernels never carry it.
        for (int32_t const visionHeadDim : {256, 512})
        {
            EXPECT_TRUE(CuteDslFMHAV2Runner::canImplement(
                8, 2, visionHeadDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kVISION_BLOCK));
            EXPECT_FALSE(CuteDslFMHAV2Runner::canImplementPaged(
                8, 2, visionHeadDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kVISION_BLOCK));
        }
        EXPECT_FALSE(
            CuteDslFMHAV2Runner::canImplement(8, 2, 512, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kPADDING));
        EXPECT_FALSE(CuteDslFMHAV2Runner::canImplementPaged(
            8, 2, 256, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kPADDING));
    }

    EXPECT_FALSE(
        CuteDslFMHAV2Runner::canImplementPaged(8, 2, 64, 103, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
    EXPECT_FALSE(
        CuteDslFMHAV2Runner::canImplementPaged(8, 2, 72, 121, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
    EXPECT_FALSE(
        CuteDslFMHAV2Runner::canImplementPaged(8, 2, 64, 121, DataType::kFLOAT, CuteDslFMHAV2MaskType::kCAUSAL));
    EXPECT_FALSE(
        CuteDslFMHAV2Runner::canImplementPaged(7, 2, 64, 121, DataType::kHALF, CuteDslFMHAV2MaskType::kCAUSAL));
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

void TestContextAttentionDenoisePaddingVarlen(std::vector<int32_t> const& qLens, std::vector<int32_t> const& kvLens)
{
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    int32_t constexpr headSize = 256;
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

    CuteDslFMHAV2Runner runner(numQHeads, numKVHeads, headSize, batchSize, seqLenQ, seqLenK);
    ASSERT_TRUE(runner.runPadding(qTensor.rawPointer(), kTensor.rawPointer(), vTensor.rawPointer(),
        oTensor.rawPointer(), cuQTensor.dataPointer<int32_t>(), cuKVTensor.dataPointer<int32_t>(), nullptr,
        1.0F / std::sqrt(static_cast<float>(headSize))));
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
}
