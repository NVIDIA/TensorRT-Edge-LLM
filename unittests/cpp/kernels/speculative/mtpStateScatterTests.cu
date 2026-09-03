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

#include "kernels/speculative/mtpStateScatterKernels.h"
#include "runtime/mambaCacheManager.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace trt_edgellm::kernel;
namespace rt = trt_edgellm::rt;

namespace
{

constexpr float kSentinel = -999.0f;

inline float refRecurrent(int32_t layer, int32_t batch, int32_t step, int32_t elem)
{
    return static_cast<float>(layer * 1000000 + batch * 10000 + step * 100 + (elem % 100));
}

inline float refConv(int32_t layer, int32_t batch, int32_t step, int32_t elem)
{
    return static_cast<float>(layer * 1000 + batch * 100 + step * 10 + (elem % 10));
}

template <typename T>
T* uploadVec(std::vector<T> const& host)
{
    T* dev = nullptr;
    cudaMalloc(&dev, host.size() * sizeof(T));
    cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    return dev;
}

} // anonymous namespace

// ============================================================================
// FP32 Recurrent State Scatter Tests (batched across layers)
// ============================================================================

class MTPStateScatterRecurrentTest : public ::testing::Test
{
protected:
    /// Run the batched recurrent scatter and verify per-(layer, batch).
    /// stateElements must be divisible by 8.  acceptLengths is 1-based (matching eagleAccept).
    void runTest(int32_t numLayers, int32_t batchSize, int32_t verifyTreeSize, int32_t stateElements,
        std::vector<int32_t> const& acceptLengthsHost)
    {
        ASSERT_EQ(static_cast<int32_t>(acceptLengthsHost.size()), batchSize);
        ASSERT_EQ(stateElements % 8, 0) << "stateElements must be divisible by 8";

        int64_t const srcPerLayer = static_cast<int64_t>(batchSize) * verifyTreeSize * stateElements;
        int64_t const dstPerLayer = static_cast<int64_t>(batchSize) * stateElements;

        // Allocate per-layer device buffers, populate src.
        std::vector<float*> dstPerLayerDev(numLayers);
        std::vector<float*> srcPerLayerDev(numLayers);
        std::vector<std::vector<float>> hSrc(numLayers, std::vector<float>(srcPerLayer));
        std::vector<std::vector<float>> hDst(numLayers, std::vector<float>(dstPerLayer, kSentinel));

        std::vector<MtpLayerInfo> hostInfos(numLayers);
        for (int32_t L = 0; L < numLayers; ++L)
        {
            for (int32_t b = 0; b < batchSize; ++b)
            {
                for (int32_t t = 0; t < verifyTreeSize; ++t)
                {
                    for (int32_t e = 0; e < stateElements; ++e)
                    {
                        int64_t const idx = (static_cast<int64_t>(b) * verifyTreeSize + t) * stateElements + e;
                        hSrc[L][idx] = refRecurrent(L, b, t, e);
                    }
                }
            }
            cudaMalloc(&srcPerLayerDev[L], srcPerLayer * sizeof(float));
            cudaMalloc(&dstPerLayerDev[L], dstPerLayer * sizeof(float));
            cudaMemcpy(srcPerLayerDev[L], hSrc[L].data(), srcPerLayer * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dstPerLayerDev[L], hDst[L].data(), dstPerLayer * sizeof(float), cudaMemcpyHostToDevice);

            // The recurrent variant only reads recurrentDst/recurrentSrc; conv pointers are unused.
            hostInfos[L] = {dstPerLayerDev[L], srcPerLayerDev[L], nullptr, nullptr};
        }

        // Upload MtpLayerInfo array + acceptLengths.
        MtpLayerInfo* dInfos = uploadVec(hostInfos);
        int32_t* dAccept = uploadVec(acceptLengthsHost);

        mtpScatterRecurrentStates(dInfos, numLayers, batchSize, verifyTreeSize, stateElements, dAccept, nullptr);
        cudaDeviceSynchronize();

        // Read back and verify.
        for (int32_t L = 0; L < numLayers; ++L)
        {
            std::vector<float> result(dstPerLayer);
            cudaMemcpy(result.data(), dstPerLayerDev[L], dstPerLayer * sizeof(float), cudaMemcpyDeviceToHost);

            for (int32_t b = 0; b < batchSize; ++b)
            {
                int32_t const acceptLen = acceptLengthsHost[b];
                int32_t const step = acceptLen - 1;
                int64_t const dstBase = static_cast<int64_t>(b) * stateElements;

                if (step < 0 || step >= verifyTreeSize - 1)
                {
                    for (int32_t e = 0; e < stateElements; ++e)
                    {
                        EXPECT_EQ(result[dstBase + e], kSentinel) << "L=" << L << " b=" << b << " e=" << e;
                    }
                }
                else
                {
                    for (int32_t e = 0; e < stateElements; ++e)
                    {
                        EXPECT_EQ(result[dstBase + e], refRecurrent(L, b, step, e))
                            << "L=" << L << " b=" << b << " step=" << step << " e=" << e;
                    }
                }
            }
            cudaFree(srcPerLayerDev[L]);
            cudaFree(dstPerLayerDev[L]);
        }
        cudaFree(dInfos);
        cudaFree(dAccept);
    }
};

TEST_F(MTPStateScatterRecurrentTest, SingleLayer_PartialReject)
{
    runTest(/*numLayers=*/1, /*batchSize=*/1, /*verifyTreeSize=*/4, /*stateElements=*/128, /*acceptLengths=*/{1});
}

TEST_F(MTPStateScatterRecurrentTest, SingleLayer_AllAccept)
{
    runTest(1, 1, 4, 128, {4});
}

TEST_F(MTPStateScatterRecurrentTest, SingleLayer_Skip)
{
    runTest(1, 1, 4, 128, {0});
}

TEST_F(MTPStateScatterRecurrentTest, SingleLayer_MixedBatch)
{
    runTest(1, 4, 4, 256, {2, 4, 0, 3});
}

// 4 layers, 2 batches — verifies layer indexing via blockIdx.y.
TEST_F(MTPStateScatterRecurrentTest, MultiLayer)
{
    runTest(4, 2, 4, 128, {2, 1});
}

// Large state: vectorized path with vecCount > blockDim (grid z > 1).
TEST_F(MTPStateScatterRecurrentTest, MultiLayer_LargeState)
{
    runTest(3, 2, 4, 4096, {2, 3});
}

// Regression: src buffer allocated with maxAlloc rows per batch but the plugin packed
// data with a smaller verifyTreeSize stride. The kernel must use the verifyTreeSize
// argument (not the buffer's allocation size) when computing the per-batch offset,
// otherwise bs >= 1 reads garbage.
//
// Layout: srcBuf[b * verifyTreeSize * stateElements + step * stateElements] = data
//         (rest of the [maxAlloc - verifyTreeSize] rows per batch is sentinel and unread)
TEST(MTPStateScatterRecurrentPackedStrideTest, BatchedReadStride)
{
    constexpr int32_t numLayers = 2;
    constexpr int32_t batchSize = 2;
    constexpr int32_t verifyTreeSize = 4;
    constexpr int32_t maxAlloc = 8; // > verifyTreeSize on purpose
    constexpr int32_t stateElements = 128;

    std::vector<int32_t> acceptLengthsHost = {2, 3}; // step=1 / step=2

    int64_t const allocPerLayer = static_cast<int64_t>(batchSize) * maxAlloc * stateElements;
    int64_t const dstPerLayer = static_cast<int64_t>(batchSize) * stateElements;

    std::vector<float*> dstDev(numLayers);
    std::vector<float*> srcDev(numLayers);
    std::vector<MtpLayerInfo> hostInfos(numLayers);

    for (int32_t L = 0; L < numLayers; ++L)
    {
        // Initialise the entire src allocation with sentinel; only fill the packed region.
        std::vector<float> hSrc(allocPerLayer, kSentinel);
        for (int32_t b = 0; b < batchSize; ++b)
        {
            for (int32_t t = 0; t < verifyTreeSize; ++t)
            {
                for (int32_t e = 0; e < stateElements; ++e)
                {
                    int64_t const idx = (static_cast<int64_t>(b) * verifyTreeSize + t) * stateElements + e;
                    hSrc[idx] = refRecurrent(L, b, t, e);
                }
            }
        }
        std::vector<float> hDst(dstPerLayer, kSentinel);

        cudaMalloc(&srcDev[L], allocPerLayer * sizeof(float));
        cudaMalloc(&dstDev[L], dstPerLayer * sizeof(float));
        cudaMemcpy(srcDev[L], hSrc.data(), allocPerLayer * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dstDev[L], hDst.data(), dstPerLayer * sizeof(float), cudaMemcpyHostToDevice);

        hostInfos[L] = {dstDev[L], srcDev[L], nullptr, nullptr};
    }

    MtpLayerInfo* dInfos = uploadVec(hostInfos);
    int32_t* dAccept = uploadVec(acceptLengthsHost);

    // Pass verifyTreeSize = verifyTreeSize (NOT maxAlloc).
    mtpScatterRecurrentStates(dInfos, numLayers, batchSize, verifyTreeSize, stateElements, dAccept, nullptr);
    cudaDeviceSynchronize();

    for (int32_t L = 0; L < numLayers; ++L)
    {
        std::vector<float> result(dstPerLayer);
        cudaMemcpy(result.data(), dstDev[L], dstPerLayer * sizeof(float), cudaMemcpyDeviceToHost);
        for (int32_t b = 0; b < batchSize; ++b)
        {
            int32_t const step = acceptLengthsHost[b] - 1;
            int64_t const dstBase = static_cast<int64_t>(b) * stateElements;
            for (int32_t e = 0; e < stateElements; ++e)
            {
                EXPECT_EQ(result[dstBase + e], refRecurrent(L, b, step, e))
                    << "L=" << L << " b=" << b << " step=" << step << " e=" << e;
            }
        }
        cudaFree(srcDev[L]);
        cudaFree(dstDev[L]);
    }
    cudaFree(dInfos);
    cudaFree(dAccept);
}

// ============================================================================
// FP16 Conv State Scatter Tests (batched across layers)
// ============================================================================

class MTPStateScatterConvTest : public ::testing::Test
{
protected:
    void runTest(int32_t numLayers, int32_t batchSize, int32_t verifyTreeSize, int32_t stateElements,
        std::vector<int32_t> const& acceptLengthsHost)
    {
        ASSERT_EQ(static_cast<int32_t>(acceptLengthsHost.size()), batchSize);
        ASSERT_EQ(stateElements % 8, 0) << "stateElements must be divisible by 8";

        int64_t const srcPerLayer = static_cast<int64_t>(batchSize) * verifyTreeSize * stateElements;
        int64_t const dstPerLayer = static_cast<int64_t>(batchSize) * stateElements;

        __half const hSentinel = __float2half(kSentinel);

        std::vector<__half*> dstPerLayerDev(numLayers);
        std::vector<__half*> srcPerLayerDev(numLayers);
        std::vector<std::vector<__half>> hSrc(numLayers, std::vector<__half>(srcPerLayer));
        std::vector<std::vector<__half>> hDst(numLayers, std::vector<__half>(dstPerLayer, hSentinel));

        std::vector<MtpLayerInfo> hostInfos(numLayers);
        for (int32_t L = 0; L < numLayers; ++L)
        {
            for (int32_t b = 0; b < batchSize; ++b)
            {
                for (int32_t t = 0; t < verifyTreeSize; ++t)
                {
                    for (int32_t e = 0; e < stateElements; ++e)
                    {
                        int64_t const idx = (static_cast<int64_t>(b) * verifyTreeSize + t) * stateElements + e;
                        hSrc[L][idx] = __float2half(static_cast<float>(L * 1000 + b * 100 + t * 10 + (e % 10)));
                    }
                }
            }
            cudaMalloc(&srcPerLayerDev[L], srcPerLayer * sizeof(__half));
            cudaMalloc(&dstPerLayerDev[L], dstPerLayer * sizeof(__half));
            cudaMemcpy(srcPerLayerDev[L], hSrc[L].data(), srcPerLayer * sizeof(__half), cudaMemcpyHostToDevice);
            cudaMemcpy(dstPerLayerDev[L], hDst[L].data(), dstPerLayer * sizeof(__half), cudaMemcpyHostToDevice);

            // The conv variant only reads convDst/convSrc; recurrent pointers are unused.
            hostInfos[L] = {nullptr, nullptr, dstPerLayerDev[L], srcPerLayerDev[L]};
        }

        MtpLayerInfo* dInfos = uploadVec(hostInfos);
        int32_t* dAccept = uploadVec(acceptLengthsHost);

        mtpScatterConvStates(dInfos, numLayers, batchSize, verifyTreeSize, stateElements, dAccept, nullptr);
        cudaDeviceSynchronize();

        for (int32_t L = 0; L < numLayers; ++L)
        {
            std::vector<__half> result(dstPerLayer);
            cudaMemcpy(result.data(), dstPerLayerDev[L], dstPerLayer * sizeof(__half), cudaMemcpyDeviceToHost);

            for (int32_t b = 0; b < batchSize; ++b)
            {
                int32_t const acceptLen = acceptLengthsHost[b];
                int32_t const step = acceptLen - 1;
                int64_t const dstBase = static_cast<int64_t>(b) * stateElements;

                if (step < 0 || step >= verifyTreeSize - 1)
                {
                    for (int32_t e = 0; e < stateElements; ++e)
                    {
                        EXPECT_EQ(__half2float(result[dstBase + e]), __half2float(hSentinel))
                            << "L=" << L << " b=" << b << " e=" << e;
                    }
                }
                else
                {
                    int64_t const srcBase = (static_cast<int64_t>(b) * verifyTreeSize + step) * stateElements;
                    for (int32_t e = 0; e < stateElements; ++e)
                    {
                        EXPECT_EQ(__half2float(result[dstBase + e]), __half2float(hSrc[L][srcBase + e]))
                            << "L=" << L << " b=" << b << " step=" << step << " e=" << e;
                    }
                }
            }
            cudaFree(srcPerLayerDev[L]);
            cudaFree(dstPerLayerDev[L]);
        }
        cudaFree(dInfos);
        cudaFree(dAccept);
    }
};

TEST_F(MTPStateScatterConvTest, SingleLayer_PartialReject)
{
    runTest(/*numLayers=*/1, /*batchSize=*/2, /*verifyTreeSize=*/4, /*stateElements=*/256, /*acceptLengths=*/{2, 1});
}

TEST_F(MTPStateScatterConvTest, SingleLayer_AllAccept)
{
    runTest(1, 2, 4, 256, {4, 4});
}

// 3 layers, 4 batches, mixed accept.
TEST_F(MTPStateScatterConvTest, MultiLayer_Mixed)
{
    runTest(3, 4, 2, 16384, {1, 2, 1, 0});
}

TEST(MambaCacheManagerAcceptedTreeScatterTest, UsesLastNonNegativeAcceptedNodeId)
{
    constexpr int32_t kNumLayers = 2;
    constexpr int32_t kBatchSize = 3;
    constexpr int32_t kVerifyTreeSize = 6;
    constexpr int32_t kMaxAcceptLen = 4;
    constexpr int32_t kRecurrentHeads = 2;
    constexpr int32_t kRecurrentHeadDim = 2;
    constexpr int32_t kRecurrentStateSize = 8;
    constexpr int32_t kConvDim = 8;
    constexpr int32_t kConvKernel = 2;
    constexpr int32_t kRecurrentElements = kRecurrentHeads * kRecurrentHeadDim * kRecurrentStateSize;
    constexpr int32_t kConvElements = kConvDim * kConvKernel;

    rt::MambaCacheManager::Config cfg{};
    cfg.numRecurrentLayers = kNumLayers;
    cfg.maxBatchSize = kBatchSize;
    cfg.recurrentStateNumHeads = kRecurrentHeads;
    cfg.recurrentStateHeadDim = kRecurrentHeadDim;
    cfg.recurrentStateSize = kRecurrentStateSize;
    cfg.convDim = kConvDim;
    cfg.convKernel = kConvKernel;
    cfg.maxIntermediateSeqLen = kVerifyTreeSize;
    cfg.recurrentStateType = nvinfer1::DataType::kFLOAT;
    cfg.convStateType = nvinfer1::DataType::kHALF;

    rt::MambaCacheManager mgr(cfg, nullptr);
    mgr.reshapeIntermediateStates(kBatchSize, kVerifyTreeSize);

    int64_t const recSrcSize = static_cast<int64_t>(kBatchSize) * kVerifyTreeSize * kRecurrentElements;
    int64_t const recDstSize = static_cast<int64_t>(kBatchSize) * kRecurrentElements;
    int64_t const convSrcSize = static_cast<int64_t>(kBatchSize) * kVerifyTreeSize * kConvElements;
    int64_t const convDstSize = static_cast<int64_t>(kBatchSize) * kConvElements;

    __half const hSentinel = __float2half(kSentinel);
    for (int32_t layer = 0; layer < kNumLayers; ++layer)
    {
        std::vector<float> hRecSrc(recSrcSize);
        std::vector<float> hRecDst(recDstSize, kSentinel);
        std::vector<__half> hConvSrc(convSrcSize);
        std::vector<__half> hConvDst(convDstSize, hSentinel);

        for (int32_t b = 0; b < kBatchSize; ++b)
        {
            for (int32_t node = 0; node < kVerifyTreeSize; ++node)
            {
                for (int32_t e = 0; e < kRecurrentElements; ++e)
                {
                    int64_t const idx = (static_cast<int64_t>(b) * kVerifyTreeSize + node) * kRecurrentElements + e;
                    hRecSrc[idx] = refRecurrent(layer, b, node, e);
                }
                for (int32_t e = 0; e < kConvElements; ++e)
                {
                    int64_t const idx = (static_cast<int64_t>(b) * kVerifyTreeSize + node) * kConvElements + e;
                    hConvSrc[idx] = __float2half(refConv(layer, b, node, e));
                }
            }
        }

        cudaMemcpy(mgr.getIntermediateRecurrentState(layer).rawPointer(), hRecSrc.data(), recSrcSize * sizeof(float),
            cudaMemcpyHostToDevice);
        cudaMemcpy(mgr.getRecurrentState(layer).rawPointer(), hRecDst.data(), recDstSize * sizeof(float),
            cudaMemcpyHostToDevice);
        cudaMemcpy(mgr.getIntermediateConvState(layer).rawPointer(), hConvSrc.data(), convSrcSize * sizeof(__half),
            cudaMemcpyHostToDevice);
        cudaMemcpy(mgr.getConvState(layer).rawPointer(), hConvDst.data(), convDstSize * sizeof(__half),
            cudaMemcpyHostToDevice);
    }

    rt::Tensor acceptedNodeIds(
        {kBatchSize, kMaxAcceptLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptedNodeIds");
    rt::Tensor acceptLengths({kBatchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "acceptLengths");

    std::vector<int32_t> hAcceptedNodeIds{
        0, 5, -1, -1, // batch 0: bonus token at the end, scatter node 5
        3, -1, 1, 4,  // batch 1: acceptLength excludes stale tail node 4, scatter node 1
        -1, -1, 0, 0  // batch 2: no accepted verify node in range, leave state unchanged
    };
    std::vector<int32_t> hAcceptLengths{4, 3, 2};
    cudaMemcpy(acceptedNodeIds.rawPointer(), hAcceptedNodeIds.data(), hAcceptedNodeIds.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice);
    cudaMemcpy(acceptLengths.rawPointer(), hAcceptLengths.data(), hAcceptLengths.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice);

    mgr.scatterAcceptedTreeStates(acceptedNodeIds, acceptLengths, nullptr);
    cudaDeviceSynchronize();

    std::vector<int32_t> const expectedNodeIds{5, 1, -1};
    for (int32_t layer = 0; layer < kNumLayers; ++layer)
    {
        std::vector<float> recResult(recDstSize);
        std::vector<__half> convResult(convDstSize);
        cudaMemcpy(recResult.data(), mgr.getRecurrentState(layer).rawPointer(), recDstSize * sizeof(float),
            cudaMemcpyDeviceToHost);
        cudaMemcpy(convResult.data(), mgr.getConvState(layer).rawPointer(), convDstSize * sizeof(__half),
            cudaMemcpyDeviceToHost);

        for (int32_t b = 0; b < kBatchSize; ++b)
        {
            int64_t const recDstBase = static_cast<int64_t>(b) * kRecurrentElements;
            int64_t const convDstBase = static_cast<int64_t>(b) * kConvElements;
            int32_t const expectedNodeId = expectedNodeIds[b];
            for (int32_t e = 0; e < kRecurrentElements; ++e)
            {
                float const expected = expectedNodeId >= 0 ? refRecurrent(layer, b, expectedNodeId, e) : kSentinel;
                EXPECT_EQ(recResult[recDstBase + e], expected) << "layer=" << layer << " batch=" << b << " elem=" << e;
            }
            for (int32_t e = 0; e < kConvElements; ++e)
            {
                float const expected
                    = expectedNodeId >= 0 ? refConv(layer, b, expectedNodeId, e) : __half2float(hSentinel);
                EXPECT_EQ(__half2float(convResult[convDstBase + e]), expected)
                    << "layer=" << layer << " batch=" << b << " elem=" << e;
            }
        }
    }
}
