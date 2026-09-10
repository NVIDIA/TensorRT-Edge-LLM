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
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "references.h"
#include "testUtils.h"

#include "common/cudaMacros.h"
#include "common/pagedKvTypes.h"

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;

struct AttnParams
{
    int32_t numQHeads;
    int32_t numKVHeads;
    int32_t headDim;
    int32_t rotaryDim;
};

constexpr int32_t kPageSize = rt::kTOKENS_PER_PAGE;

int32_t getMaxPagesPerSeq(int32_t const kvCacheCapacity)
{
    return (kvCacheCapacity + kPageSize - 1) / kPageSize;
}

int64_t pagedKvIndex(int32_t const cachePlane, int32_t const page, int32_t const token, int32_t const head,
    int32_t const dim, int32_t const numPages, int32_t const numKVHeads, int32_t const headDim)
{
    return (((static_cast<int64_t>(cachePlane) * numPages + page) * kPageSize + token) * numKVHeads + head) * headDim
        + dim;
}

std::vector<int32_t> makeIdentityPageTable(int32_t const batchSize, int32_t const maxPagesPerSeq)
{
    std::vector<int32_t> pageTable(static_cast<size_t>(batchSize) * 2 * maxPagesPerSeq);
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int32_t pageIdx = 0; pageIdx < maxPagesPerSeq; ++pageIdx)
        {
            int32_t const physicalPage = batchIdx * maxPagesPerSeq + pageIdx;
            pageTable[(batchIdx * 2 + 0) * maxPagesPerSeq + pageIdx] = physicalPage;
            pageTable[(batchIdx * 2 + 1) * maxPagesPerSeq + pageIdx] = physicalPage + batchSize * maxPagesPerSeq;
        }
    }
    return pageTable;
}

void TestRopeWriteKvPrefill(int32_t const batchSize, AttnParams const& attnParams, int32_t const kvCacheCapacity,
    int32_t const qSeqLen, float ropeTheta = 10000.0f, int32_t cosSinCacheBatchSize = 1, int32_t cosSinCacheSeqLen = 0,
    bool const enableFp8Check = false)
{
    cudaStream_t stream{nullptr};

    int32_t const headDim = attnParams.headDim;
    int32_t const rotaryDim = attnParams.rotaryDim;
    int32_t const numQHeads = attnParams.numQHeads;
    int32_t const numKVHeads = attnParams.numKVHeads;
    int32_t const maxPagesPerSeq = getMaxPagesPerSeq(kvCacheCapacity);
    int32_t const numPages = batchSize * maxPagesPerSeq;

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
    rt::Tensor kvCacheTensor(
        rt::Coords{2, numPages, kPageSize, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    std::vector<int32_t> const pageTableHost = makeIdentityPageTable(batchSize, maxPagesPerSeq);
    rt::Tensor pageTableTensor(
        rt::Coords{batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(pageTableTensor, pageTableHost);

    launchApplyRopeWriteKV(cosSinCacheTensor, std::nullopt, qTensor, kTensor, vTensor, kvCacheTensor, 1.0f, 1.0f,
        stream, true, pageTableTensor.dataPointer<int32_t>(), maxPagesPerSeq);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const qOut = copyDeviceToHost<half>(qTensor);
    auto const kOut = copyDeviceToHost<half>(kTensor);
    auto const vOut = copyDeviceToHost<half>(vTensor);
    auto const kvCacheOut = copyDeviceToHost<half>(kvCacheTensor);
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
                    int32_t const page = i * maxPagesPerSeq + j / kPageSize;
                    half const kCacheVal = kvCacheOut[pagedKvIndex(
                        /*cachePlane=*/0, page, j % kPageSize, hkv, d, numPages, numKVHeads, headDim)];
                    half const kRefVal = kReference[kvOffset + d];
                    half const vVal = vOut[kvOffset + d];
                    half const vCacheVal = kvCacheOut[pagedKvIndex(
                        /*cachePlane=*/1, page, j % kPageSize, hkv, d, numPages, numKVHeads, headDim)];
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
        rt::Tensor kvFp8(
            rt::Coords{2, numPages, kPageSize, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8);
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
                        int32_t const page = b * maxPagesPerSeq + j / kPageSize;
                        float const fk = std::fabs(__half2float(kvCacheOut[pagedKvIndex(
                            /*cachePlane=*/0, page, j % kPageSize, hkv, d, numPages, numKVHeads, headDim)]));
                        float const fv = std::fabs(__half2float(kvCacheOut[pagedKvIndex(
                            /*cachePlane=*/1, page, j % kPageSize, hkv, d, numPages, numKVHeads, headDim)]));
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
            kScaleQuantOrig, vScaleQuantOrig, stream, true, pageTableTensor.dataPointer<int32_t>(), maxPagesPerSeq);
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
                        int32_t const page = b * maxPagesPerSeq + j / kPageSize;
                        int64_t const kIdx = pagedKvIndex(
                            /*cachePlane=*/0, page, j % kPageSize, hkv, d, numPages, numKVHeads, headDim);
                        int64_t const vIdx = pagedKvIndex(
                            /*cachePlane=*/1, page, j % kPageSize, hkv, d, numPages, numKVHeads, headDim);
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
    int32_t const maxPagesPerSeq = getMaxPagesPerSeq(kvCacheCapacity);
    int32_t const numPages = batchSize * maxPagesPerSeq;

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

    // Q/K/V tensor has layout [B, S, Hq/Hkv, D].
    rt::Tensor qTensor(
        rt::Coords{batchSize, qLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor kTensor(
        rt::Coords{batchSize, qLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor vTensor(
        rt::Coords{batchSize, qLen, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor kvCacheTensor(
        rt::Coords{2, numPages, kPageSize, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    std::vector<int32_t> const pageTableHost = makeIdentityPageTable(batchSize, maxPagesPerSeq);
    rt::Tensor pageTableTensor(
        rt::Coords{batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(pageTableTensor, pageTableHost);

    // Q/K/V input will be initialized later in the loop computing the reference output.
    std::vector<half> qInput;
    std::vector<half> kInput;
    std::vector<half> vInput;

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
            stream, false, pageTableTensor.dataPointer<int32_t>(), maxPagesPerSeq);
    }
    else
    {
        launchApplyRopeWriteKVTreeDecoding(cosSinCacheTensor, seqLensTensor, customSeqLensTensor, qTensor, kTensor,
            vTensor, kvCacheTensor, 1.0f, 1.0f, stream, pageTableTensor.dataPointer<int32_t>(), maxPagesPerSeq);
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
                    int32_t const page = b * maxPagesPerSeq + inCacheIdx / kPageSize;
                    half const kVal = kvCacheOut[pagedKvIndex(
                        /*cachePlane=*/0, page, inCacheIdx % kPageSize, hkv, d, numPages, numKVHeads, headDim)];
                    half const kRefVal = kReference[kvRefOffset + d];
                    ASSERT_TRUE(isclose(kVal, kRefVal, 1e-3, 4e-3));
                    half const vVal = kvCacheOut[pagedKvIndex(
                        /*cachePlane=*/1, page, inCacheIdx % kPageSize, hkv, d, numPages, numKVHeads, headDim)];
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

        rt::Tensor kvFp8(
            rt::Coords{2, numPages, kPageSize, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8);

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
                        int32_t const page = b * maxPagesPerSeq + inCacheIdx / kPageSize;
                        float const fk = std::fabs(__half2float(kvCacheOut[pagedKvIndex(
                            /*cachePlane=*/0, page, inCacheIdx % kPageSize, hkv, d, numPages, numKVHeads, headDim)]));
                        float const fv = std::fabs(__half2float(kvCacheOut[pagedKvIndex(
                            /*cachePlane=*/1, page, inCacheIdx % kPageSize, hkv, d, numPages, numKVHeads, headDim)]));
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
                kScaleQuantOrig, vScaleQuantOrig, stream, false, pageTableTensor.dataPointer<int32_t>(),
                maxPagesPerSeq);
        }
        else
        {
            launchApplyRopeWriteKVTreeDecoding(cosSinCacheTensor, seqLensTensor, customSeqLensTensor, qTensorForFP8,
                kTensorForFP8, vTensorForFP8, kvFp8, kScaleQuantOrig, vScaleQuantOrig, stream,
                pageTableTensor.dataPointer<int32_t>(), maxPagesPerSeq);
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
                        int32_t const page = b * maxPagesPerSeq + inCacheIdx / kPageSize;
                        int64_t const kIdx = pagedKvIndex(
                            /*cachePlane=*/0, page, inCacheIdx % kPageSize, hkv, d, numPages, numKVHeads, headDim);
                        int64_t const vIdx = pagedKvIndex(
                            /*cachePlane=*/1, page, inCacheIdx % kPageSize, hkv, d, numPages, numKVHeads, headDim);
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
    int32_t const maxPagesPerSeq = getMaxPagesPerSeq(kvCacheCapacity);
    int32_t const numPages = batchSize * maxPagesPerSeq;

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

    rt::Tensor kvCacheTensor(
        rt::Coords{2, numPages, kPageSize, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    std::vector<int32_t> const pageTableHost = makeIdentityPageTable(batchSize, maxPagesPerSeq);
    rt::Tensor pageTableTensor(
        rt::Coords{batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(pageTableTensor, pageTableHost);
    rt::Tensor cosSinCacheTensor(
        rt::Coords{cosSinCacheBatchSize, kvCacheCapacity, rotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice(cosSinCacheTensor, cosSinCache);

    cudaStream_t stream{nullptr};

    auto launchPrefill = [&]() {
        launchApplyRopeWriteKV(cosSinCacheTensor, std::nullopt, qTensor, kTensor, vTensor, kvCacheTensor, 1.0f, 1.0f,
            stream, true, pageTableTensor.dataPointer<int32_t>(), maxPagesPerSeq);
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
        rt::Coords{2, numPages, kPageSize, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFP8);

    auto launchPrefillFp8 = [&]() {
        launchApplyRopeWriteKV(cosSinCacheTensor, std::nullopt, qTensor, kTensor, vTensor, kvCacheTensorFp8, 1.0f, 1.0f,
            stream, true, pageTableTensor.dataPointer<int32_t>(), maxPagesPerSeq);
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

//! Fused qk_norm + RoPE on the packed-QKV kernel (launchApplyRopeFromPackedToSplit).
//! Covers non-power-of-2 lane counts (headDim=96/80 -> ghost-lane padding) and tail tokens
//! (totalNumTokens not a multiple of tokens-per-CTA), which must join the warp collectives
//! without storing anything. Reference: per-head RMSNorm + this file's RoPE reference.
void TestRopePackedFusedNorm(int32_t const batchSize, AttnParams const& attnParams, int32_t const kvCacheCapacity,
    int32_t const qSeqLen, bool const scramblePages = false)
{
    cudaStream_t stream{nullptr};

    int32_t const headDim = attnParams.headDim;
    int32_t const rotaryDim = attnParams.rotaryDim;
    int32_t const numQHeads = attnParams.numQHeads;
    int32_t const numKVHeads = attnParams.numKVHeads;
    int32_t const combinedHeads = numQHeads + 2 * numKVHeads;
    int32_t const maxPagesPerSeq = getMaxPagesPerSeq(kvCacheCapacity);
    int32_t const numPages = batchSize * maxPagesPerSeq;
    float const rmsEps = 1e-6f;
    bool const permuteRope = true;

    std::vector<float> cosSinCache(static_cast<size_t>(kvCacheCapacity) * rotaryDim);
    uniformFloatInitialization(cosSinCache, -1, 1);
    rt::Tensor cosSinCacheTensor(
        rt::Coords{1, kvCacheCapacity, rotaryDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    copyHostToDevice(cosSinCacheTensor, cosSinCache);

    std::vector<half> qGamma(headDim);
    std::vector<half> kGamma(headDim);
    uniformFloatInitialization(qGamma, 0.5f, 1.5f);
    uniformFloatInitialization(kGamma, 0.5f, 1.5f);
    rt::Tensor qGammaTensor(rt::Coords{headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor kGammaTensor(rt::Coords{headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    copyHostToDevice(qGammaTensor, qGamma);
    copyHostToDevice(kGammaTensor, kGamma);

    std::vector<half> packedInput(static_cast<size_t>(batchSize) * qSeqLen * combinedHeads * headDim);
    uniformFloatInitialization(packedInput);
    rt::Tensor packedTensor(
        rt::Coords{batchSize, qSeqLen, combinedHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    copyHostToDevice(packedTensor, packedInput);

    rt::Tensor qScratchTensor(
        rt::Coords{batchSize, qSeqLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    // Sentinel-fill the KV cache so stray writes (tail tokens / ghost lanes) are detectable.
    half const sentinel = __float2half(777.f);
    std::vector<half> kvCacheInit(static_cast<size_t>(2) * numPages * kPageSize * numKVHeads * headDim, sentinel);
    rt::Tensor kvCacheTensor(
        rt::Coords{2, numPages, kPageSize, numKVHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    copyHostToDevice(kvCacheTensor, kvCacheInit);
    std::vector<int32_t> pageTableHost = makeIdentityPageTable(batchSize, maxPagesPerSeq);
    if (scramblePages)
    {
        for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int32_t pageIdx = 0; pageIdx < maxPagesPerSeq; ++pageIdx)
            {
                int32_t const logicalPage = batchIdx * maxPagesPerSeq + pageIdx;
                int32_t const physicalPage = numPages - 1 - logicalPage;
                pageTableHost[(batchIdx * 2 + 0) * maxPagesPerSeq + pageIdx] = physicalPage;
                pageTableHost[(batchIdx * 2 + 1) * maxPagesPerSeq + pageIdx] = physicalPage + numPages;
            }
        }
    }
    rt::Tensor pageTableTensor(
        rt::Coords{batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(pageTableTensor, pageTableHost);

    launchApplyRopeFromPackedToSplit(cosSinCacheTensor, std::nullopt, std::nullopt, packedTensor, qScratchTensor,
        kvCacheTensor, 1.0f, 1.0f, stream, pageTableTensor.dataPointer<int32_t>(), maxPagesPerSeq, nullptr, nullptr,
        nullptr, 1.0f, qGammaTensor.dataPointer<half>(), kGammaTensor.dataPointer<half>(), rmsEps);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const qOut = copyDeviceToHost<half>(qScratchTensor);
    auto const kvCacheOut = copyDeviceToHost<half>(kvCacheTensor);

    // Host reference: RMSNorm (float accumulate, half rounding) then RoPE.
    auto rmsNormHead = [&](half const* src, std::vector<half> const& gamma) {
        // Replicate the kernel's summation order (per-lane fmaf slices + butterfly
        // combine) so sumSq is bit-equal — a flat sequential sum differs by ~1 ulp
        // and can flip the pre-gamma half rounding past the test tolerance.
        int32_t constexpr kVecSize = 8;
        int32_t const lanes = headDim / kVecSize;
        int32_t paddedLanes = 1;
        while (paddedLanes < lanes)
        {
            paddedLanes <<= 1;
        }
        std::vector<float> partial(paddedLanes, 0.f);
        for (int32_t l = 0; l < lanes; ++l)
        {
            for (int32_t i = 0; i < kVecSize; ++i)
            {
                float const v = __half2float(src[l * kVecSize + i]);
                partial[l] = std::fmaf(v, v, partial[l]);
            }
        }
        for (int32_t off = paddedLanes / 2; off > 0; off >>= 1)
        {
            std::vector<float> combined(paddedLanes);
            for (int32_t l = 0; l < paddedLanes; ++l)
            {
                combined[l] = partial[l] + partial[l ^ off];
            }
            partial = std::move(combined);
        }
        float const sumSq = partial[0];
        float const invRms = 1.f / std::sqrt(sumSq / static_cast<float>(headDim) + rmsEps);
        std::vector<half> normed(headDim);
        for (int32_t d = 0; d < headDim; ++d)
        {
            // Kernel order: fp32 scale, round to half, then half-precision gamma
            // multiply (exact via fp32 — an 11x11-bit product fits fp32).
            half const scaledH = __float2half(__half2float(src[d]) * invRms);
            normed[d] = __float2half(__half2float(scaledH) * __half2float(gamma[d]));
        }
        return normed;
    };

    for (int32_t i = 0; i < batchSize; ++i)
    {
        for (int32_t j = 0; j < qSeqLen; ++j)
        {
            int64_t const tokenBase = (static_cast<int64_t>(i) * qSeqLen + j) * combinedHeads * headDim;
            int32_t const cosSinOffset = j * rotaryDim;
            auto const cosVec = std::vector<float>(
                cosSinCache.begin() + cosSinOffset, cosSinCache.begin() + cosSinOffset + rotaryDim / 2);
            auto const sinVec = std::vector<float>(
                cosSinCache.begin() + cosSinOffset + rotaryDim / 2, cosSinCache.begin() + cosSinOffset + rotaryDim);

            for (int32_t hq = 0; hq < numQHeads; ++hq)
            {
                auto const normed = rmsNormHead(packedInput.data() + tokenBase + hq * headDim, qGamma);
                auto const qRef = ropeRefCosSin(normed, 1, headDim, rotaryDim, cosVec, sinVec, permuteRope);
                int64_t const qOffset = (static_cast<int64_t>(i) * qSeqLen + j) * numQHeads * headDim + hq * headDim;
                for (int32_t d = 0; d < headDim; ++d)
                {
                    ASSERT_TRUE(isclose(qOut[qOffset + d], qRef[d], 1e-3, 1e-3))
                        << "Q mismatch b=" << i << " s=" << j << " h=" << hq << " d=" << d;
                }
            }
            for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
            {
                auto const normed = rmsNormHead(packedInput.data() + tokenBase + (numQHeads + hkv) * headDim, kGamma);
                auto const kRef = ropeRefCosSin(normed, 1, headDim, rotaryDim, cosVec, sinVec, permuteRope);
                int64_t const vSrcBase = tokenBase + (numQHeads + numKVHeads + hkv) * headDim;
                for (int32_t d = 0; d < headDim; ++d)
                {
                    int32_t const logicalPage = j / kPageSize;
                    int32_t const kPage = pageTableHost[(i * 2 + 0) * maxPagesPerSeq + logicalPage];
                    int32_t const vPage = pageTableHost[(i * 2 + 1) * maxPagesPerSeq + logicalPage] - numPages;
                    ASSERT_TRUE(
                        isclose(kvCacheOut[pagedKvIndex(
                                    /*cachePlane=*/0, kPage, j % kPageSize, hkv, d, numPages, numKVHeads, headDim)],
                            kRef[d], 1e-3, 1e-3))
                        << "K cache mismatch b=" << i << " s=" << j << " h=" << hkv << " d=" << d;
                    ASSERT_TRUE(
                        isclose(kvCacheOut[pagedKvIndex(
                                    /*cachePlane=*/1, vPage, j % kPageSize, hkv, d, numPages, numKVHeads, headDim)],
                            packedInput[vSrcBase + d], 1e-5, 1e-5))
                        << "V cache mismatch b=" << i << " s=" << j << " h=" << hkv << " d=" << d;
                }
            }
        }
        // Cache slots past the written sequence must keep the sentinel: tail tokens and
        // ghost lanes participate in the warp collectives but must not store anything.
        for (int32_t hkv = 0; hkv < numKVHeads; ++hkv)
        {
            for (int32_t slot = qSeqLen; slot < kvCacheCapacity; ++slot)
            {
                for (int32_t d = 0; d < headDim; ++d)
                {
                    int32_t const logicalPage = slot / kPageSize;
                    int32_t const kPage = pageTableHost[(i * 2 + 0) * maxPagesPerSeq + logicalPage];
                    int32_t const vPage = pageTableHost[(i * 2 + 1) * maxPagesPerSeq + logicalPage] - numPages;
                    ASSERT_TRUE(__half2float(kvCacheOut[pagedKvIndex(
                                    /*cachePlane=*/0, kPage, slot % kPageSize, hkv, d, numPages, numKVHeads, headDim)])
                            == 777.f
                        && __half2float(kvCacheOut[pagedKvIndex(
                               /*cachePlane=*/1, vPage, slot % kPageSize, hkv, d, numPages, numKVHeads, headDim)])
                            == 777.f)
                        << "Stray KV-cache write b=" << i << " h=" << hkv << " slot=" << slot << " d=" << d;
                }
            }
        }
    }

    std::cout << "TestRopePackedFusedNorm BatchSize: " << batchSize << " QHeadNum: " << numQHeads
              << " KVHeadNum: " << numKVHeads << " HeadSize: " << headDim << " RotaryDim: " << rotaryDim
              << " qSeqLen: " << qSeqLen << std::endl;
}

TEST(RopePackedRaggedPrefill, SkipsPaddingBeforePagedWrite)
{
    cudaStream_t stream{nullptr};
    int32_t constexpr batchSize = 2;
    int32_t constexpr qSeqLen = 128;
    int32_t constexpr numQHeads = 4;
    int32_t constexpr numKVHeads = 1;
    int32_t constexpr headDim = 64;
    int32_t constexpr combinedHeads = numQHeads + 2 * numKVHeads;
    int32_t constexpr kvCacheCapacity = 256;
    int32_t constexpr maxPagesPerSeq = 2;
    int32_t constexpr numPages = batchSize * maxPagesPerSeq;

    rt::Tensor cosSinCacheTensor(
        rt::Coords{1, kvCacheCapacity, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    initializeNormalRopeCosSin(
        cosSinCacheTensor.dataPointer<float>(), 10000.0F, 1.0F, 1.0F, headDim, kvCacheCapacity, stream);

    std::vector<half> packedInput(static_cast<size_t>(batchSize) * qSeqLen * combinedHeads * headDim);
    uniformFloatInitialization(packedInput);
    rt::Tensor packedTensor(
        rt::Coords{batchSize, qSeqLen, combinedHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    copyHostToDevice(packedTensor, packedInput);

    rt::Tensor qScratchTensor(
        rt::Coords{batchSize, qSeqLen, numQHeads, headDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    half const sentinel = __float2half(777.0F);
    std::vector<half> kvCacheInit(
        static_cast<size_t>(2 * numPages) * rt::kTOKENS_PER_PAGE * numKVHeads * headDim, sentinel);
    rt::Tensor kvCacheTensor(rt::Coords{2, numPages, rt::kTOKENS_PER_PAGE, numKVHeads, headDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    copyHostToDevice(kvCacheTensor, kvCacheInit);

    // Batch 0 has one live token at the final cache slot while batch 1 forces
    // the physical Q extent to 128. Padding rows would index page-table row 2
    // and RoPE positions beyond 255 unless cuQSeqLens is applied first.
    rt::Tensor kvCacheEndLensTensor({batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(kvCacheEndLensTensor, std::vector<int32_t>{kvCacheCapacity - 1 + qSeqLen, qSeqLen});
    rt::Tensor cuQSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(cuQSeqLensTensor, std::vector<int32_t>{0, 1, qSeqLen + 1});

    // Only leased logical pages are mapped. K and V use independent absolute
    // flattened page IDs; every unused slot is -1.
    rt::Tensor pageTableTensor(
        rt::Coords{batchSize, 2, maxPagesPerSeq}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    copyHostToDevice(pageTableTensor, std::vector<int32_t>{-1, 1, -1, 5, 2, -1, 6, -1});

    launchApplyRopeFromPackedToSplit(cosSinCacheTensor, rt::OptionalInputTensor{kvCacheEndLensTensor},
        rt::OptionalInputTensor{}, packedTensor, qScratchTensor, kvCacheTensor, 1.0F, 1.0F, stream,
        pageTableTensor.dataPointer<int32_t>(), maxPagesPerSeq, nullptr, nullptr, nullptr, 1.0F, nullptr, nullptr,
        1e-6F, rt::OptionalInputTensor{cuQSeqLensTensor});
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    auto const qOut = copyDeviceToHost<half>(qScratchTensor);
    for (int32_t row = 1; row < qSeqLen; ++row)
    {
        for (int32_t head = 0; head < numQHeads; ++head)
        {
            for (int32_t dim = 0; dim < headDim; ++dim)
            {
                size_t const idx = (static_cast<size_t>(row) * numQHeads + static_cast<size_t>(head)) * headDim + dim;
                EXPECT_EQ(__half2float(qOut[idx]), 0.0F);
            }
        }
    }

    auto const kvOut = copyDeviceToHost<half>(kvCacheTensor);
    for (int32_t page : {0, 3, 4, 7})
    {
        size_t const begin = static_cast<size_t>(page) * rt::kTOKENS_PER_PAGE * numKVHeads * headDim;
        size_t const end = begin + static_cast<size_t>(rt::kTOKENS_PER_PAGE) * numKVHeads * headDim;
        for (size_t idx = begin; idx < end; ++idx)
        {
            EXPECT_EQ(__half2float(kvOut[idx]), 777.0F) << "Unexpected write to unleased physical page " << page;
        }
    }
}

TEST(RopePackedFusedNorm, Accuracy)
{
    // Power-of-2 lane count baseline (headDim=128 -> 16 lanes, no ghosts); odd seq len for
    // tail tokens (2*7=14 tokens, 8 tokens/CTA -> the second CTA carries 2 tail rows).
    TestRopePackedFusedNorm(2, {8, 2, 128, 128}, 16, 7);
    // headDim=96 -> 12 lanes padded to 16 (4 ghost lanes per head), non-XOR-able RoPE
    // partner (gmem-reload path), plus tail tokens (2*5=10 tokens, 8 tokens/CTA).
    TestRopePackedFusedNorm(2, {4, 2, 96, 96}, 16, 5);
    // headDim=80 -> 10 lanes padded to 16 (6 ghost lanes per head).
    TestRopePackedFusedNorm(1, {4, 2, 80, 80}, 16, 3);
}

TEST(RopePackedFusedNorm, ScrambledPageTableSpansPageBoundary)
{
    TestRopePackedFusedNorm(2, {4, 2, 64, 64}, 256, 130, /*scramblePages=*/true);
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
