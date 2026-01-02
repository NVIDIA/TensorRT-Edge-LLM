/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "references.h"
#include "testUtils.h"

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

struct AttnParams
{
    uint32_t numQHeads;
    uint32_t numKVHeads;
    uint32_t headDim;
    uint32_t rotaryDim;
};

void TestRopeWriteKvPrefill(uint32_t const batchSize, AttnParams const& attnParams, int32_t const kvCacheCapacity,
    int32_t const qSeqLen, float ropeTheta = 10000.0f, int32_t cosSinCacheBatchSize = 1, int32_t cosSinCacheSeqLen = 0)
{
    cudaStream_t stream{nullptr};

    uint32_t const headDim = attnParams.headDim;
    uint32_t const rotaryDim = attnParams.rotaryDim;
    uint32_t const numQHeads = attnParams.numQHeads;
    uint32_t const numKVHeads = attnParams.numKVHeads;
    int32_t const kvCacheVolume = batchSize * (numKVHeads + numKVHeads) * kvCacheCapacity * headDim;

    assert(cosSinCacheBatchSize == 1 || cosSinCacheBatchSize == batchSize);
    if (cosSinCacheSeqLen == 0)
    {
        cosSinCacheSeqLen = kvCacheCapacity;
    }

    std::vector<half> qkvInput;
    std::vector<half> qkvReference;

    bool const permuteRope = true;
    float const ropeScale = 1.0f;
    rt::Tensor cosSinCacheTensor(rt::Coords{cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT);
    int64_t const cosSinCacheVolume = cosSinCacheTensor.getShape().volume();
    std::vector<float> cosSinCache(cosSinCacheVolume);
    bool const useRegularRope = cosSinCacheBatchSize == 1 && rotaryDim % 64 == 0;
    if (useRegularRope)
    {
        // Initialize normal CosSinCache to real values.
        initializeNormalRopeCosSin(
            cosSinCacheTensor.dataPointer<float>(), ropeTheta, ropeScale, rotaryDim, kvCacheCapacity, stream);
    }
    else
    {
        // Random initialize CosSinCache for non-64-multiple rotaryDim or cosSinCacheBatchSize != 1.
        uniformFloatInitialization(cosSinCache, -1, 1);
        CUDA_CHECK(cudaMemcpy(cosSinCacheTensor.rawPointer(), cosSinCache.data(), cosSinCacheVolume * sizeof(float),
            cudaMemcpyHostToDevice));
    }

    for (int32_t i = 0; i < batchSize; i++)
    {
        for (int32_t j = 0; j < qSeqLen; j++)
        {
            std::vector<half> qij(numQHeads * headDim);
            std::vector<half> kij(numKVHeads * headDim);
            std::vector<half> vij(numKVHeads * headDim);

            uniformFloatInitialization(qij);
            uniformFloatInitialization(kij);
            uniformFloatInitialization(vij);
            // QKV input has layout of [B, S, H, D]

            qkvInput.insert(qkvInput.end(), qij.begin(), qij.end());
            qkvInput.insert(qkvInput.end(), kij.begin(), kij.end());
            qkvInput.insert(qkvInput.end(), vij.begin(), vij.end());

            std::vector<half> qRoped;
            std::vector<half> kRoped;
            if (useRegularRope)
            {
                qRoped = ropeRef(qij, numQHeads, headDim, rotaryDim, j, ropeScale, ropeTheta, permuteRope);
                kRoped = ropeRef(kij, numKVHeads, headDim, rotaryDim, j, ropeScale, ropeTheta, permuteRope);
            }
            else
            {
                // Calculate the correct batch index for cosSinCache
                int32_t const cosSinCacheBatchIdx = (cosSinCacheBatchSize == 1) ? 0 : i;
                int32_t const cosSinCacheOffset = cosSinCacheBatchIdx * cosSinCacheSeqLen * rotaryDim + j * rotaryDim;
                auto const cosVec = std::vector<float>(
                    cosSinCache.begin() + cosSinCacheOffset, cosSinCache.begin() + cosSinCacheOffset + rotaryDim / 2);
                auto const sinVec = std::vector<float>(cosSinCache.begin() + cosSinCacheOffset + rotaryDim / 2,
                    cosSinCache.begin() + cosSinCacheOffset + rotaryDim);

                qRoped = ropeRefCosSin(qij, numQHeads, headDim, rotaryDim, cosVec, sinVec, permuteRope);
                kRoped = ropeRefCosSin(kij, numKVHeads, headDim, rotaryDim, cosVec, sinVec, permuteRope);
            }

            qkvReference.insert(qkvReference.end(), qRoped.begin(), qRoped.end());
            qkvReference.insert(qkvReference.end(), kRoped.begin(), kRoped.end());
            qkvReference.insert(qkvReference.end(), vij.begin(), vij.end());
        }
    }

    rt::Tensor qkvTensor(rt::Coords{batchSize, qSeqLen, numQHeads + numKVHeads * 2, headDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    CUDA_CHECK(
        cudaMemcpy(qkvTensor.rawPointer(), qkvInput.data(), qkvInput.size() * sizeof(half), cudaMemcpyHostToDevice));
    rt::Tensor kvCacheTensor(rt::Coords{batchSize, 2, numKVHeads, kvCacheCapacity, headDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);

    // Set qOut, kvCacheStartIds, tokenPosIds to nullptr since they are not used in prefill case.
    launchApplyRopeWriteKVPackedQKV(cosSinCacheTensor, qkvTensor, kvCacheTensor, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<half> qkvOut(qkvTensor.getShape().volume());
    CUDA_CHECK(cudaMemcpy(
        qkvOut.data(), qkvTensor.rawPointer(), qkvTensor.getShape().volume() * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<half> kvCacheOut(kvCacheTensor.getShape().volume());
    CUDA_CHECK(cudaMemcpy(kvCacheOut.data(), kvCacheTensor.rawPointer(),
        kvCacheTensor.getShape().volume() * sizeof(half), cudaMemcpyDeviceToHost));

    KvCacheIndexer kvIndexer(batchSize, numKVHeads, kvCacheCapacity, headDim);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        int32_t const batchOffset = i * qSeqLen * (numQHeads + 2 * numKVHeads) * headDim;
        for (int32_t j = 0; j < qSeqLen; ++j)
        {
            int32_t const tokenOffset = j * (numQHeads + 2 * numKVHeads) * headDim;
            for (int32_t hq = 0; hq < numQHeads; ++hq)
            {
                int32_t const qOffset = batchOffset + tokenOffset + hq * headDim;
                for (int32_t d = 0; d < headDim; ++d)
                {
                    half const qVal = qkvOut[qOffset + d];
                    half const qRefVal = qkvReference[qOffset + d];
                    ASSERT_TRUE(isclose(qVal, qRefVal, 1e-3, 1e-3));
                }
            }
            for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
            {
                int32_t const kOffset = batchOffset + tokenOffset + numQHeads * headDim + hkv * headDim;
                int32_t const vOffset
                    = batchOffset + tokenOffset + numQHeads * headDim + numKVHeads * headDim + hkv * headDim;
                for (int32_t d = 0; d < headDim; ++d)
                {
                    half const kVal = qkvOut[kOffset + d];
                    half const kCacheVal = kvCacheOut[kvIndexer.indexK(i, hkv, j, d)];
                    half const kRefVal = qkvReference[kOffset + d];
                    half const vVal = qkvOut[vOffset + d];
                    half const vCacheVal = kvCacheOut[kvIndexer.indexV(i, hkv, j, d)];
                    half const vRefVal = qkvReference[vOffset + d];
                    ASSERT_TRUE(isclose(kVal, kRefVal, 1e-3, 1e-3));
                    ASSERT_TRUE(isclose(vVal, vRefVal, 1e-3, 1e-3));
                    ASSERT_TRUE(isclose(kCacheVal, kVal, 1e-5, 1e-5));
                    ASSERT_TRUE(isclose(vCacheVal, vVal, 1e-5, 1e-5));
                }
            }
        }
    }

    std::cout << "TestRopeWriteKvPrefill "
              << "BatchSize: " << batchSize << " QHeadNum: " << numQHeads << " KVHeadNum: " << numKVHeads
              << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim << " KVCacheCapacity: " << kvCacheCapacity
              << " qSeqLen: " << qSeqLen << " cosSinCacheBatchSize: " << cosSinCacheBatchSize
              << " cosSinCacheSeqLen: " << cosSinCacheSeqLen << std::endl;
}

void TestRopeWriteKvDecode(int32_t const batchSize, AttnParams const& attnParams, int32_t const kvCacheCapacity,
    int32_t const qLen, float ropeTheta = 10000.0f, bool const isTreeAttention = false,
    int32_t cosSinCacheBatchSize = 1)
{
    // Not tested for MROPE which supply positional encoding coefficients as input tensor.
    EXPECT_TRUE(qLen == 1 || isTreeAttention);
    EXPECT_TRUE(cosSinCacheBatchSize == 1 || cosSinCacheBatchSize == batchSize);
    // We will randomly initialize KVCache length with smallest value of kvCacheCapacity / 4.
    EXPECT_TRUE(kvCacheCapacity > 4 * qLen);
    cudaStream_t stream{nullptr};

    uint32_t const headDim = attnParams.headDim;
    uint32_t const rotaryDim = attnParams.rotaryDim;
    uint32_t const numQHeads = attnParams.numQHeads;
    uint32_t const numKVHeads = attnParams.numKVHeads;
    int32_t const cosSinCacheSeqLen = kvCacheCapacity;

    // QKV tensor has layout [B, S, Hq+Hk+Hv, D]. KV cache has layout [B, 2, S, Hkv, D].
    rt::Tensor qkvTensor(rt::Coords{batchSize, qLen, numQHeads + numKVHeads * 2, headDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    rt::Tensor kvCacheTensor(rt::Coords{batchSize, 2, numKVHeads, kvCacheCapacity, headDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    int64_t const kvCacheVolume = kvCacheTensor.getShape().volume();

    // QKV input will be initialized later in the loop computing the reference output.
    std::vector<half> qkvInput;
    std::vector<half> kvCache(kvCacheVolume, 0);

    // Reference output of Q, K, V all have layout [B, S, H, D].
    std::vector<half> qreference;
    std::vector<half> kreference;
    std::vector<half> vreference;

    // Random initialized the total length which is committed kv-cache length + new tokens length.
    std::vector<int32_t> fullSeqLens(batchSize);
    uniformIntInitialization(fullSeqLens, kvCacheCapacity / 4, kvCacheCapacity);
    std::vector<int32_t> customSeqLens;

    bool const permuteRope = true;
    float const ropeScale = 1.0f;
    rt::Tensor cosSinCacheTensor(rt::Coords{cosSinCacheBatchSize, cosSinCacheSeqLen, rotaryDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT);
    int64_t const cosSinCacheVolume = cosSinCacheTensor.getShape().volume();
    std::vector<float> cosSinCache(cosSinCacheVolume);
    bool const useRegularRope = cosSinCacheBatchSize == 1 && rotaryDim % 64 == 0;
    if (useRegularRope)
    { // Initialize normal CosSinCache to real values.
        initializeNormalRopeCosSin(
            cosSinCacheTensor.dataPointer<float>(), ropeTheta, ropeScale, rotaryDim, kvCacheCapacity, stream);
    }
    else
    { // Random initialize CosSinCache for non-64-multiple rotaryDim or cosSinCacheBatchSize != 1.
        uniformFloatInitialization(cosSinCache, -1, 1);
        CUDA_CHECK(cudaMemcpy(cosSinCacheTensor.rawPointer(), cosSinCache.data(), cosSinCacheVolume * sizeof(float),
            cudaMemcpyHostToDevice));
    }

    for (int32_t i = 0; i < batchSize; i++)
    {
        int32_t const qStartIdx = fullSeqLens[i] - qLen;
        // With speculative decoding, the sequence index is not identical to kvcache index.
        std::vector<int32_t> customSeqLen(qLen);
        uniformIntInitialization(customSeqLen, qStartIdx, qStartIdx + qLen - 1);
        customSeqLens.insert(customSeqLens.end(), customSeqLen.begin(), customSeqLen.end());

        for (int32_t j = 0; j < qLen; j++)
        {
            std::vector<half> qi(numQHeads * headDim);
            std::vector<half> ki(numKVHeads * headDim);
            std::vector<half> vi(numKVHeads * headDim);

            uniformFloatInitialization(qi);
            uniformFloatInitialization(ki);
            uniformFloatInitialization(vi);

            qkvInput.insert(qkvInput.end(), qi.begin(), qi.end());
            qkvInput.insert(qkvInput.end(), ki.begin(), ki.end());
            qkvInput.insert(qkvInput.end(), vi.begin(), vi.end());

            int32_t seqIdx = qStartIdx + j;
            if (isTreeAttention)
            {
                // Pick custom sequence index if tree attention is enabled.
                seqIdx = customSeqLen[j];
            }

            std::vector<half> qRefij;
            std::vector<half> kRefij;
            if (useRegularRope)
            {
                qRefij = ropeRef(qi, numQHeads, headDim, rotaryDim, seqIdx, ropeScale, ropeTheta, permuteRope);
                kRefij = ropeRef(ki, numKVHeads, headDim, rotaryDim, seqIdx, ropeScale, ropeTheta, permuteRope);
            }
            else
            {
                // Calculate the correct batch index for cosSinCache
                int32_t const cosSinCacheBatchIdx = (cosSinCacheBatchSize == 1) ? 0 : i;
                int32_t const cosSinCacheOffset
                    = cosSinCacheBatchIdx * cosSinCacheSeqLen * rotaryDim + seqIdx * rotaryDim;

                auto const cosVec = std::vector<float>(
                    cosSinCache.begin() + cosSinCacheOffset, cosSinCache.begin() + cosSinCacheOffset + rotaryDim / 2);
                auto const sinVec = std::vector<float>(cosSinCache.begin() + cosSinCacheOffset + rotaryDim / 2,
                    cosSinCache.begin() + cosSinCacheOffset + rotaryDim);

                qRefij = ropeRefCosSin(qi, numQHeads, headDim, rotaryDim, cosVec, sinVec, permuteRope);
                kRefij = ropeRefCosSin(ki, numKVHeads, headDim, rotaryDim, cosVec, sinVec, permuteRope);
            }

            qreference.insert(qreference.end(), qRefij.begin(), qRefij.end());
            kreference.insert(kreference.end(), kRefij.begin(), kRefij.end());
            vreference.insert(vreference.end(), vi.begin(), vi.end());
        }
    }

    CUDA_CHECK(
        cudaMemcpy(qkvTensor.rawPointer(), qkvInput.data(), qkvInput.size() * sizeof(half), cudaMemcpyHostToDevice));
    rt::Tensor seqLensTensor(rt::Coords{batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    CUDA_CHECK(cudaMemcpy(
        seqLensTensor.rawPointer(), fullSeqLens.data(), fullSeqLens.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    rt::Tensor customSeqLensTensor(rt::Coords{batchSize, qLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    CUDA_CHECK(cudaMemcpy(customSeqLensTensor.rawPointer(), customSeqLens.data(),
        customSeqLens.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Output Q tensor.
    rt::Tensor qOutTensor(
        rt::Coords{batchSize, qLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    if (!isTreeAttention)
    {
        launchApplyRopeWriteKVContinuousQAndKVCache(
            cosSinCacheTensor, seqLensTensor, qkvTensor, kvCacheTensor, qOutTensor, stream);
    }
    else
    {
        launchApplyRopeWriteKVTreeDecoding(
            cosSinCacheTensor, seqLensTensor, customSeqLensTensor, qkvTensor, kvCacheTensor, qOutTensor, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<half> qOut(qOutTensor.getShape().volume());
    CUDA_CHECK(cudaMemcpy(
        qOut.data(), qOutTensor.rawPointer(), qOutTensor.getShape().volume() * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<half> kvCacheOut(kvCacheTensor.getShape().volume());
    CUDA_CHECK(cudaMemcpy(kvCacheOut.data(), kvCacheTensor.rawPointer(),
        kvCacheTensor.getShape().volume() * sizeof(half), cudaMemcpyDeviceToHost));

    // Directly compare the output of Q since output and reference have the same layout.
    EXPECT_EQ(qOut.size(), qreference.size());
    for (int32_t i = 0; i < qOut.size(); ++i)
    {
        ASSERT_TRUE(isclose(qOut[i], qreference[i], 1e-3, 4e-3));
    }

    KvCacheIndexer kvIndexer(batchSize, numKVHeads, kvCacheCapacity, headDim);

    for (int32_t b = 0; b < batchSize; ++b)
    {
        int32_t const qStartIdx = fullSeqLens[b] - qLen;
        for (int32_t s = 0; s < qLen; ++s)
        {
            int32_t const inCacheIdx = qStartIdx + s;
            for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
            {
                int32_t const kvRefOffset = b * qLen * numKVHeads * headDim + s * numKVHeads * headDim + hkv * headDim;
                for (int32_t d = 0; d < headDim; ++d)
                {
                    half const kVal = kvCacheOut[kvIndexer.indexK(b, hkv, inCacheIdx, d)];
                    half const kRefVal = kreference[kvRefOffset + d];
                    ASSERT_TRUE(isclose(kVal, kRefVal, 1e-3, 4e-3));
                    half const vVal = kvCacheOut[kvIndexer.indexV(b, hkv, inCacheIdx, d)];
                    half const vRefVal = vreference[kvRefOffset + d];
                    ASSERT_TRUE(isclose(vVal, vRefVal, 1e-3, 4e-3));
                }
            }
        }
    }

    std::cout << "TestRopeWriteKvDecode "
              << "BatchSize: " << batchSize << " QHeadNum: " << numQHeads << " KVHeadNum: " << numKVHeads
              << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim << " KVCacheCapacity: " << kvCacheCapacity
              << " QLength: " << qLen << " Total Sequence Lengths (including past KVcache): " << fullSeqLens
              << " RopeScale: " << ropeScale << " RopeTheta: " << ropeTheta
              << " cosSinCacheBatchSize: " << cosSinCacheBatchSize << " cosSinCacheSeqLen: " << cosSinCacheSeqLen
              << std::endl;
}

void BenchmarkRopeWriteKv(
    uint32_t const batchSize, AttnParams const& attnParams, int32_t const qSeqLen, int32_t cosSinCacheBatchSize = 1)
{
    uint32_t const headDim = attnParams.headDim;
    uint32_t const rotaryDim = attnParams.rotaryDim;
    uint32_t const numQHeads = attnParams.numQHeads;
    uint32_t const numKVHeads = attnParams.numKVHeads;
    int32_t const kvCacheCapacity = 1024 + qSeqLen;

    std::vector<half> qkvInput(batchSize * qSeqLen * (numQHeads + 2 * numKVHeads) * headDim);
    assert(cosSinCacheBatchSize == 1 || cosSinCacheBatchSize == batchSize);
    std::vector<float> cosSinCache(cosSinCacheBatchSize * kvCacheCapacity * rotaryDim);

    // Initialize the data to non-zero values to avoid the benchmark data is non-realistic.
    uniformFloatInitialization(cosSinCache, -1, 1);
    uniformFloatInitialization(qkvInput);

    rt::Tensor qkvTensor(rt::Coords{batchSize, qSeqLen, numQHeads + numKVHeads * 2, headDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    CUDA_CHECK(
        cudaMemcpy(qkvTensor.rawPointer(), qkvInput.data(), qkvInput.size() * sizeof(half), cudaMemcpyHostToDevice));
    rt::Tensor kvCacheTensor(rt::Coords{batchSize, 2, numKVHeads, kvCacheCapacity, headDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    rt::Tensor cosSinCacheTensor(
        rt::Coords{cosSinCacheBatchSize, kvCacheCapacity, rotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    CUDA_CHECK(cudaMemcpy(cosSinCacheTensor.rawPointer(), cosSinCache.data(),
        cosSinCacheTensor.getShape().volume() * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream{nullptr};
    int32_t const tokenToProcess = batchSize * qSeqLen;

    auto launchPrefill
        = [&]() { launchApplyRopeWriteKVPackedQKV(cosSinCacheTensor, qkvTensor, kvCacheTensor, stream); };

    constexpr int32_t numWarmup = 10;
    for (int32_t i = 0; i < numWarmup; i++)
    {
        launchPrefill();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int32_t numBenchIter = 100;

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launchPrefill();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsedTime{0.0f};
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Bench Perf: BatchSize: " << batchSize << " QHeadNum: " << numQHeads << " KVHeadNum: " << numKVHeads
              << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim << " qSeqLen: " << qSeqLen
              << " cosSinCacheBatchSize: " << cosSinCacheBatchSize << std::endl;
    std::cout << "RopeWriteKv(non-interleave) time: " << elapsedTime / numBenchIter << " ms" << std::endl;
}

TEST(RopeWriteKvPrefill, Accuracy)
{
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, kvCacheCapacity = 2048, qLen = 512
    TestRopeWriteKvPrefill(1, {32, 8, 128, 128}, 2048, 512);
    // QheadNum = 24, kvHeadNum = 3, headSize = 128, rotaryDim = 128, kvCacheCapacity = 4096, qLen = 512
    TestRopeWriteKvPrefill(2, {24, 3, 128, 128}, 4096, 512);
    // QheadNum = 28, kvHeadNum = 7, headSize = 128, rotaryDim = 128, kvCacheCapacity = 2048, qLen = 512
    TestRopeWriteKvPrefill(1, {28, 7, 128, 128}, 2048, 512);
    // QheadNum = 16, kvHeadNum = 4, headSize = 64, rotaryDim = 64, kvCacheCapacity = 2048, qLen = 512
    TestRopeWriteKvPrefill(4, {16, 4, 64, 64}, 2048, 512);
    // QheadNum = 24, kvHeadNum = 8, headSize = 128, rotaryDim = 96, kvCacheCapacity = 4096, qLen = 512
    TestRopeWriteKvPrefill(2, {24, 8, 128, 96}, 4096, 512);
    // QheadNum = 24, kvHeadNum = 8, headSize = 128, rotaryDim = 96, kvCacheCapacity = 4096, qLen = 512,
    // cosSinCacheBatchSize = 2, cosSinCacheSeqLen = 8192
    TestRopeWriteKvPrefill(2, {24, 8, 128, 96}, 4096, 512, 10000.0f, 2, 8192);
}

TEST(RopeWriteKvDecodeVanilla, Accuracy)
{
    // qHeadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, kvCacheCapacity = 2048, qLen = 1, isTreeAttention
    // = false
    TestRopeWriteKvDecode(1, {32, 8, 128, 128}, 2048, 1, 10000.0f, false);
    // QheadNum = 28, kvHeadNum = 4, headSize = 128, rotaryDim = 128, kvCacheCapacity = 4096, qLen = 1, isTreeAttention
    // = false
    TestRopeWriteKvDecode(1, {28, 4, 128, 128}, 4096, 1, 500000.0f, false);
    // QheadNum = 16, kvHeadNum = 2, headSize = 64, rotaryDim = 64, kvCacheCapacity = 4096, qLen = 1, isTreeAttention =
    // false
    TestRopeWriteKvDecode(1, {16, 2, 64, 64}, 4096, 1, 10000.0f, false);
    // QheadNum = 24, kvHeadNum = 4, headSize = 128, rotaryDim = 128, kvCacheCapacity = 4096, qLen = 1, isTreeAttention
    // = false
    TestRopeWriteKvDecode(1, {24, 4, 128, 128}, 4096, 1, 10000.0f, false);
    // QheadNum = 24, kvHeadNum = 8, headSize = 128, rotaryDim = 96, kvCacheCapacity = 4096, qLen = 1, isTreeAttention =
    // false
    TestRopeWriteKvDecode(2, {24, 8, 128, 96}, 4096, 1, 10000.0f, false);
    // QheadNum = 24, kvHeadNum = 8, headSize = 128, rotaryDim = 96, kvCacheCapacity = 4096, qLen = 1, isTreeAttention =
    // false, cosSinCacheBatchSize = 2, cosSinCacheSeqLen = 8192
    TestRopeWriteKvDecode(2, {24, 8, 128, 96}, 4096, 1, 10000.0f, false, 2);
}

TEST(RopeWriteKvDecodeTreeAttention, Accuracy)
{
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, kvCacheCapacity = 2048, qLen = 4, isTreeAttention
    // = true
    TestRopeWriteKvDecode(1, {32, 8, 128, 128}, 2048, 4, 10000.0f, true);
    // QheadNum = 28, kvHeadNum = 4, headSize = 128, rotaryDim = 128, kvCacheCapacity = 4096, qLen = 32, isTreeAttention
    // = true
    TestRopeWriteKvDecode(1, {28, 4, 128, 128}, 4096, 32, 500000.0f, true);
    // QheadNum = 24, kvHeadNum = 6, headSize = 64, rotaryDim = 64, kvCacheCapacity = 4096, qLen = 64, isTreeAttention =
    // true
    TestRopeWriteKvDecode(1, {24, 6, 64, 64}, 4096, 64, 10000.0f, true);
    // QheadNum = 16, kvHeadNum = 2, headSize = 128, rotaryDim = 128, kvCacheCapacity = 4096, qLen = 50, isTreeAttention
    // = true
    TestRopeWriteKvDecode(1, {16, 2, 128, 128}, 4096, 50, 10000.0f, true);
    // QheadNum = 24, kvHeadNum = 8, headSize = 128, rotaryDim = 96, kvCacheCapacity = 4096, qLen = 512, isTreeAttention
    // = true
    TestRopeWriteKvDecode(2, {24, 8, 128, 96}, 4096, 32, 10000.0f, true);
    // QheadNum = 24, kvHeadNum = 8, headSize = 128, rotaryDim = 96, kvCacheCapacity = 4096, qLen = 512, isTreeAttention
    // = true, cosSinCacheBatchSize = 2
    TestRopeWriteKvDecode(2, {24, 8, 128, 96}, 4096, 32, 10000.0f, true, 2);
}

TEST(RopeWriteKvPrefill, Benchmark)
{
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, qLen = 1024
    BenchmarkRopeWriteKv(1, {32, 8, 128, 128}, 1024);
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, qLen = 2048
    BenchmarkRopeWriteKv(2, {24, 3, 128, 128}, 2048);
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, qLen = 4096
    BenchmarkRopeWriteKv(1, {28, 7, 128, 128}, 4096);
    // QheadNum = 16, kvHeadNum = 4, headSize = 64, rotaryDim = 64, qLen = 1024
    BenchmarkRopeWriteKv(4, {16, 4, 64, 64}, 1024);
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, qLen = 512, cosSinCacheBatchSize = 2
    BenchmarkRopeWriteKv(2, {32, 8, 128, 128}, 512, 2);
}
