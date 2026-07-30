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

#include <gtest/gtest.h>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "references.h"
#include "testUtils.h"

#include "common/cudaMacros.h"

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

struct AttnParams
{
    int32_t numQHeads;
    int32_t numKVHeads;
    int32_t headDim;
    int32_t rotaryDim;
};

void TestRopeWriteKvPrefill(int32_t const batchSize, AttnParams const& attnParams, int32_t const kvCacheCapacity,
    int32_t const qSeqLen, float ropeTheta = 10000.0f, int32_t cosSinCacheBatchSize = 1, int32_t cosSinCacheSeqLen = 0,
    bool const enableFp8Check = false)
{
    cudaStream_t stream{nullptr};

    int32_t const headDim = attnParams.headDim;
    int32_t const rotaryDim = attnParams.rotaryDim;
    int32_t const numQHeads = attnParams.numQHeads;
    int32_t const numKVHeads = attnParams.numKVHeads;

    assert(cosSinCacheBatchSize == 1 || cosSinCacheBatchSize == batchSize);
    if (cosSinCacheSeqLen == 0)
    {
        cosSinCacheSeqLen = kvCacheCapacity;
    }

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
            cosSinCacheTensor.dataPointer<float>(), ropeTheta, ropeScale, 1.0F, rotaryDim, kvCacheCapacity, stream);
    }
    else
    {
        // Random initialize CosSinCache for non-64-multiple rotaryDim or cosSinCacheBatchSize != 1.
        uniformFloatInitialization(cosSinCache, -1, 1);
        copyHostToDevice(cosSinCacheTensor, cosSinCache);
    }

    std::vector<half> qInput;
    std::vector<half> kInput;
    std::vector<half> vInput;
    std::vector<half> qReference;
    std::vector<half> kReference;
    std::vector<half> vReference;
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
            // Q, K, V inputs have layout of [B, S, H, D]

            qInput.insert(qInput.end(), qij.begin(), qij.end());
            kInput.insert(kInput.end(), kij.begin(), kij.end());
            vInput.insert(vInput.end(), vij.begin(), vij.end());

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
            qReference.insert(qReference.end(), qRoped.begin(), qRoped.end());
            kReference.insert(kReference.end(), kRoped.begin(), kRoped.end());
            vReference.insert(vReference.end(), vij.begin(), vij.end());
        }
    }

    rt::Tensor qTensor(
        rt::Coords{batchSize, qSeqLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor kTensor(
        rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor vTensor(
        rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    copyHostToDevice(qTensor, qInput);
    copyHostToDevice(kTensor, kInput);
    copyHostToDevice(vTensor, vInput);
    rt::Tensor kvCacheTensor(rt::Coords{batchSize, 2, numKVHeads, kvCacheCapacity, headDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);

    launchApplyRopeWriteKV(cosSinCacheTensor, std::nullopt, qTensor, kTensor, vTensor, kvCacheTensor, 1.0f, 1.0f,
        stream, true,
        /*pageTable=*/nullptr, /*maxPagesPerSeq=*/0);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const qOut = copyDeviceToHost<half>(qTensor);
    auto const kOut = copyDeviceToHost<half>(kTensor);
    auto const vOut = copyDeviceToHost<half>(vTensor);
    auto const kvCacheOut = copyDeviceToHost<half>(kvCacheTensor);
    KvCacheIndexer kvIndexer(batchSize, numKVHeads, kvCacheCapacity, headDim);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        for (int32_t j = 0; j < qSeqLen; ++j)
        {
            for (int32_t hq = 0; hq < numQHeads; ++hq)
            {
                int32_t const qOffset = i * qSeqLen * numQHeads * headDim + j * numQHeads * headDim + hq * headDim;
                for (int32_t d = 0; d < headDim; ++d)
                {
                    half const qVal = qOut[qOffset + d];
                    half const qRefVal = qReference[qOffset + d];
                    ASSERT_TRUE(isclose(qVal, qRefVal, 1e-3, 1e-3));
                }
            }
            for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
            {
                int32_t const kvOffset = i * qSeqLen * numKVHeads * headDim + j * numKVHeads * headDim + hkv * headDim;
                for (int32_t d = 0; d < headDim; ++d)
                {
                    half const kVal = kOut[kvOffset + d];
                    half const kCacheVal = kvCacheOut[kvIndexer.indexK(i, hkv, j, d)];
                    half const kRefVal = kReference[kvOffset + d];
                    half const vVal = vOut[kvOffset + d];
                    half const vCacheVal = kvCacheOut[kvIndexer.indexV(i, hkv, j, d)];
                    half const vRefVal = vReference[kvOffset + d];
                    ASSERT_TRUE(isclose(kVal, kRefVal, 1e-3, 1e-3));
                    ASSERT_TRUE(isclose(vVal, vRefVal, 1e-3, 1e-3));
                    ASSERT_TRUE(isclose(kCacheVal, kVal, 1e-5, 1e-5));
                    ASSERT_TRUE(isclose(vCacheVal, vVal, 1e-5, 1e-5));
                }
            }
        }
    }

    std::cout << "TestRopeWriteKvPrefill [FP16 KV cache] "
              << "BatchSize: " << batchSize << " QHeadNum: " << numQHeads << " KVHeadNum: " << numKVHeads
              << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim << " KVCacheCapacity: " << kvCacheCapacity
              << " qSeqLen: " << qSeqLen << " cosSinCacheBatchSize: " << cosSinCacheBatchSize
              << " cosSinCacheSeqLen: " << cosSinCacheSeqLen << std::endl;

#if SUPPORTS_FP8
    if (enableFp8Check)
    {
        // Re-create a fresh Q/K/V tensor for the FP8 path so that RoPE is applied starting
        // from the original (unmodified) Q/K/V input, matching the FP16 reference setup.
        rt::Tensor qTensorForFP8(
            rt::Coords{batchSize, qSeqLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor kTensorForFP8(
            rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor vTensorForFP8(
            rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        copyHostToDevice(qTensorForFP8, qInput);
        copyHostToDevice(kTensorForFP8, kInput);
        copyHostToDevice(vTensorForFP8, vInput);

        // FP8 KV cache path: reuse same Q/K/V input and CosSin cache, compare KV FP8 vs FP16 (after dequant)
        rt::Tensor kvFp8(rt::Coords{batchSize, 2, numKVHeads, kvCacheCapacity, headDim}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFP8);
        // Derive a realistic FP8 scale from the *written* FP16 KV cache region (qSeqLen tokens).
        // NOTE: KV cache capacity is larger than qSeqLen; elements outside [0, qSeqLen) are not written by the kernel
        // and may be uninitialized, so we must not include them in amax/accuracy checks.
        // We map max(|K|) and max(|V|) into FP8 E4M3 finite range (separately).
        float kAmax = 0.0F;
        float vAmax = 0.0F;
        for (int32_t b = 0; b < batchSize; ++b)
        {
            for (int32_t j = 0; j < qSeqLen; ++j)
            {
                for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
                {
                    for (int32_t d = 0; d < headDim; ++d)
                    {
                        float const fk = std::fabs(__half2float(kvCacheOut[kvIndexer.indexK(b, hkv, j, d)]));
                        float const fv = std::fabs(__half2float(kvCacheOut[kvIndexer.indexV(b, hkv, j, d)]));
                        kAmax = std::max(kAmax, fk);
                        vAmax = std::max(vAmax, fv);
                    }
                }
            }
        }

        // FP8 E4M3 max finite value
        constexpr float FP8_E4M3_MAX = 448.0F;
        assert(kAmax > 0.0F && vAmax > 0.0F);
        // To avoid large scale value to cause intermittent ref check failure, limit the range of kAmax and vAmax
        // to be larger than 64.0F.
        kAmax = std::max(kAmax, 64.0F);
        vAmax = std::max(vAmax, 64.0F);
        float const kScaleQuantOrig = kAmax / FP8_E4M3_MAX;
        float const vScaleQuantOrig = vAmax / FP8_E4M3_MAX;
        float const kScaleOrigQuant = 1.0F / kScaleQuantOrig;
        float const vScaleOrigQuant = 1.0F / vScaleQuantOrig;

        launchApplyRopeWriteKV(cosSinCacheTensor, std::nullopt, qTensorForFP8, kTensorForFP8, vTensorForFP8, kvFp8,
            kScaleQuantOrig, vScaleQuantOrig, stream, true, /*pageTable=*/nullptr, /*maxPagesPerSeq=*/0);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const kvOutFp8 = copyDeviceToHost<__nv_fp8_e4m3>(kvFp8);

        for (int32_t b = 0; b < batchSize; ++b)
        {
            for (int32_t j = 0; j < qSeqLen; ++j)
            {
                for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
                {
                    for (int32_t d = 0; d < headDim; ++d)
                    {
                        size_t const kIdx = kvIndexer.indexK(b, hkv, j, d);
                        size_t const vIdx = kvIndexer.indexV(b, hkv, j, d);
                        float const kRefFp8QuantizedFp16
                            = static_cast<float>(__nv_fp8_e4m3(__half2float(kvCacheOut[kIdx]) * kScaleOrigQuant));
                        float const vRefFp8QuantizedFp16
                            = static_cast<float>(__nv_fp8_e4m3(__half2float(kvCacheOut[vIdx]) * vScaleOrigQuant));
                        float const k8 = static_cast<float>(kvOutFp8[kIdx]);
                        float const v8 = static_cast<float>(kvOutFp8[vIdx]);
                        ASSERT_TRUE(isclose(k8, kRefFp8QuantizedFp16, 1e-3, 1e-3));
                        ASSERT_TRUE(isclose(v8, vRefFp8QuantizedFp16, 1e-3, 1e-3));
                    }
                }
            }
        }

        std::cout << "TestRopeWriteKvPrefill [FP8 KV cache] "
                  << "BatchSize: " << batchSize << " QHeadNum: " << numQHeads << " KVHeadNum: " << numKVHeads
                  << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim << " KVCacheCapacity: " << kvCacheCapacity
                  << " qSeqLen: " << qSeqLen << " cosSinCacheBatchSize: " << cosSinCacheBatchSize
                  << " cosSinCacheSeqLen: " << cosSinCacheSeqLen << std::endl;
    }
#else
    (void) enableFp8Check;
#endif
}

void TestRopeWriteKvDecode(int32_t const batchSize, AttnParams const& attnParams, int32_t const kvCacheCapacity,
    int32_t const qLen, float ropeTheta = 10000.0f, bool const isTreeAttention = false,
    int32_t cosSinCacheBatchSize = 1, bool const enableFp8Check = false)
{
    // Not tested for MROPE which supply positional encoding coefficients as input tensor.
    EXPECT_TRUE(qLen == 1 || isTreeAttention);
    EXPECT_TRUE(cosSinCacheBatchSize == 1 || cosSinCacheBatchSize == batchSize);
    // We will randomly initialize KVCache length with smallest value of kvCacheCapacity / 4.
    EXPECT_TRUE(kvCacheCapacity > 4 * qLen);
    cudaStream_t stream{nullptr};

    int32_t const headDim = attnParams.headDim;
    int32_t const rotaryDim = attnParams.rotaryDim;
    int32_t const numQHeads = attnParams.numQHeads;
    int32_t const numKVHeads = attnParams.numKVHeads;
    int32_t const cosSinCacheSeqLen = kvCacheCapacity;

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
    {
        // Initialize normal CosSinCache to real values.
        initializeNormalRopeCosSin(
            cosSinCacheTensor.dataPointer<float>(), ropeTheta, ropeScale, 1.0F, rotaryDim, kvCacheCapacity, stream);
    }
    else
    {
        // Random initialize CosSinCache for non-64-multiple rotaryDim or cosSinCacheBatchSize != 1.
        uniformFloatInitialization(cosSinCache, -1, 1);
        copyHostToDevice(cosSinCacheTensor, cosSinCache);
    }

    // Q/K/V tensor has layout [B, S, Hq/Hkv, D]. KV cache has layout [B, 2, Hkv, S, D].
    rt::Tensor qTensor(
        rt::Coords{batchSize, qLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor kTensor(
        rt::Coords{batchSize, qLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor vTensor(
        rt::Coords{batchSize, qLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor kvCacheTensor(rt::Coords{batchSize, 2, numKVHeads, kvCacheCapacity, headDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    int64_t const kvCacheVolume = kvCacheTensor.getShape().volume();

    // Q/K/V input will be initialized later in the loop computing the reference output.
    std::vector<half> qInput;
    std::vector<half> kInput;
    std::vector<half> vInput;
    std::vector<half> kvCache(kvCacheVolume, __float2half(0.0f));

    // Reference output of Q, K, V all have layout [B, S, H, D].
    std::vector<half> qReference;
    std::vector<half> kReference;
    std::vector<half> vReference;

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
            qInput.insert(qInput.end(), qi.begin(), qi.end());
            kInput.insert(kInput.end(), ki.begin(), ki.end());
            vInput.insert(vInput.end(), vi.begin(), vi.end());

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

            qReference.insert(qReference.end(), qRefij.begin(), qRefij.end());
            kReference.insert(kReference.end(), kRefij.begin(), kRefij.end());
            vReference.insert(vReference.end(), vi.begin(), vi.end());
        }
    }

    copyHostToDevice(qTensor, qInput);
    copyHostToDevice(kTensor, kInput);
    copyHostToDevice(vTensor, vInput);
    rt::Tensor seqLensTensor(rt::Coords{batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(seqLensTensor, fullSeqLens);
    rt::Tensor customSeqLensTensor(rt::Coords{batchSize, qLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(customSeqLensTensor, customSeqLens);

    if (!isTreeAttention)
    {
        launchApplyRopeWriteKV(cosSinCacheTensor, seqLensTensor, qTensor, kTensor, vTensor, kvCacheTensor, 1.0f, 1.0f,
            stream, false,
            /*pageTable=*/nullptr, /*maxPagesPerSeq=*/0);
    }
    else
    {
        launchApplyRopeWriteKVTreeDecoding(cosSinCacheTensor, seqLensTensor, customSeqLensTensor, qTensor, kTensor,
            vTensor, kvCacheTensor, 1.0f, 1.0f, stream, /*pageTable=*/nullptr, /*maxPagesPerSeq=*/0);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Output Q tensor.
    auto const qOut = copyDeviceToHost<half>(qTensor);
    auto const kvCacheOut = copyDeviceToHost<half>(kvCacheTensor);

    // Directly compare the output of Q since output and reference have the same layout.
    EXPECT_EQ(qOut.size(), qReference.size());
    for (size_t i = 0; i < qOut.size(); ++i)
    {
        ASSERT_TRUE(isclose(qOut[i], qReference[i], 1e-3, 4e-3));
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
                    half const kRefVal = kReference[kvRefOffset + d];
                    ASSERT_TRUE(isclose(kVal, kRefVal, 1e-3, 4e-3));
                    half const vVal = kvCacheOut[kvIndexer.indexV(b, hkv, inCacheIdx, d)];
                    half const vRefVal = vReference[kvRefOffset + d];
                    ASSERT_TRUE(isclose(vVal, vRefVal, 1e-3, 4e-3));
                }
            }
        }
    }

    std::cout << "TestRopeWriteKvDecode [FP16 KV cache] "
              << "BatchSize: " << batchSize << " QHeadNum: " << numQHeads << " KVHeadNum: " << numKVHeads
              << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim << " KVCacheCapacity: " << kvCacheCapacity
              << " QLength: " << qLen << " Total Sequence Lengths (including past KVcache): " << fullSeqLens
              << " RopeScale: " << ropeScale << " RopeTheta: " << ropeTheta
              << " cosSinCacheBatchSize: " << cosSinCacheBatchSize << " cosSinCacheSeqLen: " << cosSinCacheSeqLen
              << std::endl;

#if SUPPORTS_FP8
    if (enableFp8Check)
    {
        // FP8 KV cache path: reuse same Q/K/V input and CosSin cache, compare KV FP8 vs FP16 (after dequant)
        rt::Tensor qTensorForFP8(
            rt::Coords{batchSize, qLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor kTensorForFP8(
            rt::Coords{batchSize, qLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor vTensorForFP8(
            rt::Coords{batchSize, qLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        copyHostToDevice(qTensorForFP8, qInput);
        copyHostToDevice(kTensorForFP8, kInput);
        copyHostToDevice(vTensorForFP8, vInput);

        rt::Tensor kvFp8(rt::Coords{batchSize, 2, numKVHeads, kvCacheCapacity, headDim}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFP8);

        float kAmax = 0.0F;
        float vAmax = 0.0F;
        for (int32_t b = 0; b < batchSize; ++b)
        {
            int32_t const qStartIdx = fullSeqLens[b] - qLen;
            for (int32_t s = 0; s < qLen; ++s)
            {
                int32_t const inCacheIdx = qStartIdx + s;
                for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
                {
                    for (int32_t d = 0; d < headDim; ++d)
                    {
                        float const fk = std::fabs(__half2float(kvCacheOut[kvIndexer.indexK(b, hkv, inCacheIdx, d)]));
                        float const fv = std::fabs(__half2float(kvCacheOut[kvIndexer.indexV(b, hkv, inCacheIdx, d)]));
                        kAmax = std::max(kAmax, fk);
                        vAmax = std::max(vAmax, fv);
                    }
                }
            }
        }

        // FP8 E4M3 max finite value
        constexpr float FP8_E4M3_MAX = 448.0F;
        assert(kAmax > 0.0F && vAmax > 0.0F);
        kAmax = std::max(kAmax, 64.0F);
        vAmax = std::max(vAmax, 64.0F);
        float const kScaleQuantOrig = kAmax / FP8_E4M3_MAX;
        float const vScaleQuantOrig = vAmax / FP8_E4M3_MAX;
        float const kScaleOrigQuant = 1.0F / kScaleQuantOrig;
        float const vScaleOrigQuant = 1.0F / vScaleQuantOrig;

        if (!isTreeAttention)
        {
            launchApplyRopeWriteKV(cosSinCacheTensor, seqLensTensor, qTensorForFP8, kTensorForFP8, vTensorForFP8, kvFp8,
                kScaleQuantOrig, vScaleQuantOrig, stream, false, /*pageTable=*/nullptr, /*maxPagesPerSeq=*/0);
        }
        else
        {
            launchApplyRopeWriteKVTreeDecoding(cosSinCacheTensor, seqLensTensor, customSeqLensTensor, qTensorForFP8,
                kTensorForFP8, vTensorForFP8, kvFp8, kScaleQuantOrig, vScaleQuantOrig, stream, /*pageTable=*/nullptr,
                /*maxPagesPerSeq=*/0);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto const kvOutFp8 = copyDeviceToHost<__nv_fp8_e4m3>(kvFp8);

        for (int32_t b = 0; b < batchSize; ++b)
        {
            int32_t const qStartIdx = fullSeqLens[b] - qLen;
            for (int32_t s = 0; s < qLen; ++s)
            {
                int32_t const inCacheIdx = qStartIdx + s;
                for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
                {
                    for (int32_t d = 0; d < headDim; ++d)
                    {
                        size_t const kIdx = kvIndexer.indexK(b, hkv, inCacheIdx, d);
                        size_t const vIdx = kvIndexer.indexV(b, hkv, inCacheIdx, d);
                        float const kRefFp8QuantizedFp16
                            = static_cast<float>(__nv_fp8_e4m3(__half2float(kvCacheOut[kIdx]) * kScaleOrigQuant));
                        float const vRefFp8QuantizedFp16
                            = static_cast<float>(__nv_fp8_e4m3(__half2float(kvCacheOut[vIdx]) * vScaleOrigQuant));
                        float const k8 = static_cast<float>(kvOutFp8[kIdx]);
                        float const v8 = static_cast<float>(kvOutFp8[vIdx]);
                        ASSERT_TRUE(isclose(k8, kRefFp8QuantizedFp16, 1e-3, 1e-3));
                        ASSERT_TRUE(isclose(v8, vRefFp8QuantizedFp16, 1e-3, 1e-3));
                    }
                }
            }
        }

        std::cout << "TestRopeWriteKvDecode [FP8 KV cache] "
                  << "BatchSize: " << batchSize << " QHeadNum: " << numQHeads << " KVHeadNum: " << numKVHeads
                  << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim << " KVCacheCapacity: " << kvCacheCapacity
                  << " QLength: " << qLen << " Total Sequence Lengths (including past KVcache): " << fullSeqLens
                  << " RopeScale: " << ropeScale << " RopeTheta: " << ropeTheta
                  << " cosSinCacheBatchSize: " << cosSinCacheBatchSize << " cosSinCacheSeqLen: " << cosSinCacheSeqLen
                  << std::endl;
    }
#else
    (void) enableFp8Check;
#endif
}

void BenchmarkRopeWriteKv(
    int32_t const batchSize, AttnParams const& attnParams, int32_t const qSeqLen, int32_t cosSinCacheBatchSize = 1)
{
    int32_t const headDim = attnParams.headDim;
    int32_t const rotaryDim = attnParams.rotaryDim;
    int32_t const numQHeads = attnParams.numQHeads;
    int32_t const numKVHeads = attnParams.numKVHeads;
    int32_t const kvCacheCapacity = 1024 + qSeqLen;

    // Initialize the data to non-zero values to avoid the benchmark data is non-realistic.
    std::vector<half> qInput(batchSize * qSeqLen * numQHeads * headDim);
    std::vector<half> kInput(batchSize * qSeqLen * numKVHeads * headDim);
    std::vector<half> vInput(batchSize * qSeqLen * numKVHeads * headDim);
    assert(cosSinCacheBatchSize == 1 || cosSinCacheBatchSize == batchSize);
    std::vector<float> cosSinCache(cosSinCacheBatchSize * kvCacheCapacity * rotaryDim);

    uniformFloatInitialization(cosSinCache, -1, 1);
    uniformFloatInitialization(qInput);
    uniformFloatInitialization(kInput);
    uniformFloatInitialization(vInput);

    rt::Tensor qTensor(
        rt::Coords{batchSize, qSeqLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor kTensor(
        rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor vTensor(
        rt::Coords{batchSize, qSeqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    copyHostToDevice(qTensor, qInput);
    copyHostToDevice(kTensor, kInput);
    copyHostToDevice(vTensor, vInput);

    rt::Tensor kvCacheTensor(rt::Coords{batchSize, 2, numKVHeads, kvCacheCapacity, headDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    rt::Tensor cosSinCacheTensor(
        rt::Coords{cosSinCacheBatchSize, kvCacheCapacity, rotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice(cosSinCacheTensor, cosSinCache);

    cudaStream_t stream{nullptr};

    auto launchPrefill = [&]() {
        launchApplyRopeWriteKV(cosSinCacheTensor, std::nullopt, qTensor, kTensor, vTensor, kvCacheTensor, 1.0f, 1.0f,
            stream, true,
            /*pageTable=*/nullptr, /*maxPagesPerSeq=*/0);
    };

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
    std::cout << "Bench Perf [FP16 KV cache]: BatchSize: " << batchSize << " QHeadNum: " << numQHeads
              << " KVHeadNum: " << numKVHeads << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim
              << " qSeqLen: " << qSeqLen << " cosSinCacheBatchSize: " << cosSinCacheBatchSize << std::endl;
    std::cout << "RopeWriteKv(non-interleave) time: " << elapsedTime / numBenchIter << " ms" << std::endl;

#if SUPPORTS_FP8
    // FP8 KV cache benchmark: reuse same Q/K/V and CosSin cache, but write KV cache in FP8 with a fixed scale of 1.0.
    rt::Tensor kvCacheTensorFp8(
        rt::Coords{batchSize, 2, numKVHeads, kvCacheCapacity, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8);

    auto launchPrefillFp8 = [&]() {
        launchApplyRopeWriteKV(cosSinCacheTensor, std::nullopt, qTensor, kTensor, vTensor, kvCacheTensorFp8, 1.0f, 1.0f,
            stream, true,
            /*pageTable=*/nullptr, /*maxPagesPerSeq=*/0);
    };

    for (int32_t i = 0; i < numWarmup; i++)
    {
        launchPrefillFp8();
    }

    cudaEventRecord(start, stream);
    for (int32_t i = 0; i < numBenchIter; i++)
    {
        launchPrefillFp8();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Bench Perf [FP8 KV cache]: BatchSize: " << batchSize << " QHeadNum: " << numQHeads
              << " KVHeadNum: " << numKVHeads << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim
              << " qSeqLen: " << qSeqLen << " cosSinCacheBatchSize: " << cosSinCacheBatchSize << std::endl;
    std::cout << "RopeWriteKv(non-interleave) FP8 time: " << elapsedTime / numBenchIter << " ms" << std::endl;
#endif
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

TEST(RopeWriteKvPrefill, AccuracyFp8)
{
    // QheadNum = 32, kvHeadNum = 8, headSize = 128, rotaryDim = 128, kvCacheCapacity = 2048, qLen = 512
    TestRopeWriteKvPrefill(1, {32, 8, 128, 128}, 2048, 512, 10000.0f, 1, 0, true);
    // QheadNum = 24, kvHeadNum = 3, headSize = 128, rotaryDim = 128, kvCacheCapacity = 4096, qLen = 512
    TestRopeWriteKvPrefill(2, {24, 3, 128, 128}, 4096, 512, 10000.0f, 1, 0, true);
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

TEST(RopeWriteKvDecodeVanilla, AccuracyFp8)
{
    // Mirror vanilla decode tests but enable FP8 KV cache verification.
    TestRopeWriteKvDecode(1, {32, 8, 128, 128}, 2048, 1, 10000.0f, false, 1, true);
    TestRopeWriteKvDecode(2, {24, 8, 128, 96}, 4096, 1, 10000.0f, false, 1, true);
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

TEST(RopeWriteKvDecodeTreeAttention, AccuracyFp8)
{
    // Mirror tree attention decode tests but enable FP8 KV cache verification.
    TestRopeWriteKvDecode(1, {32, 8, 128, 128}, 2048, 4, 10000.0f, true, 1, true);
    TestRopeWriteKvDecode(2, {24, 8, 128, 96}, 4096, 32, 10000.0f, true, 1, true);
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
    // Same benchmark shapes as FP16 benchmark, but FP8 KV-cache path is enabled.
    BenchmarkRopeWriteKv(2, {32, 8, 128, 128}, 512, 2);
}
