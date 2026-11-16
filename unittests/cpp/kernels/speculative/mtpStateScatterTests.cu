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

std::vector<int32_t> identityStateIndices(int32_t batchSize)
{
    std::vector<int32_t> indices(batchSize);
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        indices[batch] = batch;
    }
    return indices;
}

} // anonymous namespace

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
        int32_t* dStateIndices = uploadVec(identityStateIndices(batchSize));

        mtpScatterConvStates(
            dInfos, numLayers, batchSize, batchSize, verifyTreeSize, stateElements, dAccept, dStateIndices, nullptr);
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

                if (step < 0 || step >= verifyTreeSize)
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
        cudaFree(dStateIndices);
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

TEST(MTPStateScatterConvResidentSlotTest, CommitsFullAndPartialAcceptsToSelectedSlots)
{
    // Given remapped resident slots, when full and partial prefixes commit, then only their selected rows change.
    constexpr int32_t batchSize = 2;
    constexpr int32_t verifyTreeSize = 4;
    constexpr int32_t stateElements = 8;
    constexpr int32_t poolRows = 5;

    std::vector<__half> source(static_cast<size_t>(batchSize) * verifyTreeSize * stateElements);
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        for (int32_t step = 0; step < verifyTreeSize; ++step)
        {
            for (int32_t element = 0; element < stateElements; ++element)
            {
                source[(static_cast<size_t>(batch) * verifyTreeSize + step) * stateElements + element]
                    = __float2half(refConv(0, batch, step, element));
            }
        }
    }
    std::vector<__half> destination(static_cast<size_t>(poolRows) * stateElements, __float2half(kSentinel));
    std::vector<int32_t> const acceptLengths{verifyTreeSize, 2};
    std::vector<int32_t> const stateIndices{3, 1};

    __half* sourceDevice = uploadVec(source);
    __half* destinationDevice = uploadVec(destination);
    MtpLayerInfo* layerInfoDevice
        = uploadVec(std::vector<MtpLayerInfo>{{nullptr, nullptr, destinationDevice, sourceDevice}});
    int32_t* acceptLengthsDevice = uploadVec(acceptLengths);
    int32_t* stateIndicesDevice = uploadVec(stateIndices);

    mtpScatterConvStates(layerInfoDevice, 1, batchSize, poolRows, verifyTreeSize, stateElements, acceptLengthsDevice,
        stateIndicesDevice, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(destination.data(), destinationDevice, destination.size() * sizeof(__half), cudaMemcpyDeviceToHost);

    for (int32_t slot = 0; slot < poolRows; ++slot)
    {
        for (int32_t element = 0; element < stateElements; ++element)
        {
            float expected = kSentinel;
            if (slot == stateIndices[0])
            {
                expected = refConv(0, 0, verifyTreeSize - 1, element);
            }
            else if (slot == stateIndices[1])
            {
                expected = refConv(0, 1, acceptLengths[1] - 1, element);
            }
            EXPECT_EQ(__half2float(destination[static_cast<size_t>(slot) * stateElements + element]), expected)
                << "slot=" << slot << " element=" << element;
        }
    }

    cudaFree(sourceDevice);
    cudaFree(destinationDevice);
    cudaFree(layerInfoDevice);
    cudaFree(acceptLengthsDevice);
    cudaFree(stateIndicesDevice);
}

TEST(MTPStateScatterConvPackedStrideTest, BatchedReadStride)
{
    // Given packed verify rows in a larger allocation, when two batches commit, then reads use the active stride.
    constexpr int32_t numLayers = 2;
    constexpr int32_t batchSize = 2;
    constexpr int32_t verifyTreeSize = 4;
    constexpr int32_t maxAlloc = 8; // > verifyTreeSize on purpose
    constexpr int32_t stateElements = 128;

    std::vector<int32_t> acceptLengthsHost = {2, 3}; // step=1 / step=2

    int64_t const allocPerLayer = static_cast<int64_t>(batchSize) * maxAlloc * stateElements;
    int64_t const dstPerLayer = static_cast<int64_t>(batchSize) * stateElements;

    std::vector<__half*> dstDev(numLayers);
    std::vector<__half*> srcDev(numLayers);
    std::vector<MtpLayerInfo> hostInfos(numLayers);

    for (int32_t L = 0; L < numLayers; ++L)
    {
        // Initialise the entire src allocation with sentinel; only fill the packed region.
        std::vector<__half> hSrc(allocPerLayer, __float2half(kSentinel));
        for (int32_t b = 0; b < batchSize; ++b)
        {
            for (int32_t t = 0; t < verifyTreeSize; ++t)
            {
                for (int32_t e = 0; e < stateElements; ++e)
                {
                    int64_t const idx = (static_cast<int64_t>(b) * verifyTreeSize + t) * stateElements + e;
                    hSrc[idx] = __float2half(refConv(L, b, t, e));
                }
            }
        }
        std::vector<__half> hDst(dstPerLayer, __float2half(kSentinel));

        cudaMalloc(&srcDev[L], allocPerLayer * sizeof(__half));
        cudaMalloc(&dstDev[L], dstPerLayer * sizeof(__half));
        cudaMemcpy(srcDev[L], hSrc.data(), allocPerLayer * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(dstDev[L], hDst.data(), dstPerLayer * sizeof(__half), cudaMemcpyHostToDevice);

        hostInfos[L] = {nullptr, nullptr, dstDev[L], srcDev[L]};
    }

    MtpLayerInfo* dInfos = uploadVec(hostInfos);
    int32_t* dAccept = uploadVec(acceptLengthsHost);
    int32_t* dStateIndices = uploadVec(identityStateIndices(batchSize));

    // Pass verifyTreeSize = verifyTreeSize (NOT maxAlloc).
    mtpScatterConvStates(
        dInfos, numLayers, batchSize, batchSize, verifyTreeSize, stateElements, dAccept, dStateIndices, nullptr);
    cudaDeviceSynchronize();

    for (int32_t L = 0; L < numLayers; ++L)
    {
        std::vector<__half> result(dstPerLayer);
        cudaMemcpy(result.data(), dstDev[L], dstPerLayer * sizeof(__half), cudaMemcpyDeviceToHost);
        for (int32_t b = 0; b < batchSize; ++b)
        {
            int32_t const step = acceptLengthsHost[b] - 1;
            int64_t const dstBase = static_cast<int64_t>(b) * stateElements;
            for (int32_t e = 0; e < stateElements; ++e)
            {
                EXPECT_EQ(__half2float(result[dstBase + e]), refConv(L, b, step, e))
                    << "L=" << L << " b=" << b << " step=" << step << " e=" << e;
            }
        }
        cudaFree(srcDev[L]);
        cudaFree(dstDev[L]);
    }
    cudaFree(dInfos);
    cudaFree(dAccept);
    cudaFree(dStateIndices);
}

// ============================================================================
// FP16 Conv State Scatter Tests (batched across layers)
TEST(MTPStateScatterAcceptedTreeConvTest, UsesLastNonNegativeAcceptedNodeId)
{
    // Given padded accepted paths, when conv snapshots commit, then the last valid in-range node wins.
    constexpr int32_t kNumLayers = 2;
    constexpr int32_t kBatchSize = 3;
    constexpr int32_t kVerifyTreeSize = 6;
    constexpr int32_t kMaxAcceptLen = 4;
    constexpr int32_t kRecurrentHeads = 2;
    constexpr int32_t kRecurrentHeadDim = 2;
    constexpr int32_t kRecurrentStateSize = 8;
    constexpr int32_t kConvDim = 8;
    constexpr int32_t kConvKernel = 2;
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
    cfg.recurrentStateType = trt_edgellm::DataType::kFLOAT;
    cfg.convStateType = trt_edgellm::DataType::kHALF;

    rt::MambaCacheManager mgr(cfg, nullptr);
    mgr.reshapeIntermediateStates(kBatchSize, kVerifyTreeSize);

    int64_t const convSrcSize = static_cast<int64_t>(kBatchSize) * kVerifyTreeSize * kConvElements;
    int64_t const convDstSize = static_cast<int64_t>(kBatchSize) * kConvElements;

    __half const hSentinel = __float2half(kSentinel);
    for (int32_t layer = 0; layer < kNumLayers; ++layer)
    {
        std::vector<__half> hConvSrc(convSrcSize);
        std::vector<__half> hConvDst(convDstSize, hSentinel);

        for (int32_t b = 0; b < kBatchSize; ++b)
        {
            for (int32_t node = 0; node < kVerifyTreeSize; ++node)
            {
                for (int32_t e = 0; e < kConvElements; ++e)
                {
                    int64_t const idx = (static_cast<int64_t>(b) * kVerifyTreeSize + node) * kConvElements + e;
                    hConvSrc[idx] = __float2half(refConv(layer, b, node, e));
                }
            }
        }

        cudaMemcpy(mgr.getIntermediateConvState(layer).rawPointer(), hConvSrc.data(), convSrcSize * sizeof(__half),
            cudaMemcpyHostToDevice);
        cudaMemcpy(mgr.getConvState(layer).rawPointer(), hConvDst.data(), convDstSize * sizeof(__half),
            cudaMemcpyHostToDevice);
    }

    rt::Tensor acceptedNodeIds(
        {kBatchSize, kMaxAcceptLen}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "acceptedNodeIds");
    rt::Tensor acceptLengths({kBatchSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "acceptLengths");
    rt::Tensor stateIndices({kBatchSize}, rt::DeviceType::kGPU, trt_edgellm::DataType::kINT32, "stateIndices");

    std::vector<int32_t> hAcceptedNodeIds{
        0, 5, -1, -1, // batch 0: bonus token at the end, scatter node 5
        3, -1, 1, 4,  // batch 1: acceptLength excludes stale tail node 4, scatter node 1
        -1, -1, 0, 0  // batch 2: no accepted verify node in range, leave state unchanged
    };
    std::vector<int32_t> hAcceptLengths{4, 3, 2};
    std::vector<int32_t> hStateIndices{0, 1, 2};
    cudaMemcpy(acceptedNodeIds.rawPointer(), hAcceptedNodeIds.data(), hAcceptedNodeIds.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice);
    cudaMemcpy(acceptLengths.rawPointer(), hAcceptLengths.data(), hAcceptLengths.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice);
    cudaMemcpy(stateIndices.rawPointer(), hStateIndices.data(), hStateIndices.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice);

    std::vector<MtpLayerInfo> infos;
    for (int32_t layer = 0; layer < kNumLayers; ++layer)
        infos.push_back(
            {nullptr, nullptr, mgr.getConvState(layer).rawPointer(), mgr.getIntermediateConvState(layer).rawPointer()});
    MtpLayerInfo* deviceInfos = uploadVec(infos);
    mtpScatterAcceptedTreeConvStates(deviceInfos, kNumLayers, kBatchSize, kBatchSize, kVerifyTreeSize, kConvElements,
        acceptedNodeIds.dataPointer<int32_t>(), kMaxAcceptLen, acceptLengths.dataPointer<int32_t>(),
        stateIndices.dataPointer<int32_t>(), nullptr);
    cudaDeviceSynchronize();
    cudaFree(deviceInfos);

    std::vector<int32_t> const expectedNodeIds{5, 1, -1};
    for (int32_t layer = 0; layer < kNumLayers; ++layer)
    {
        std::vector<__half> convResult(convDstSize);
        cudaMemcpy(convResult.data(), mgr.getConvState(layer).rawPointer(), convDstSize * sizeof(__half),
            cudaMemcpyDeviceToHost);

        for (int32_t b = 0; b < kBatchSize; ++b)
        {
            int64_t const convDstBase = static_cast<int64_t>(b) * kConvElements;
            int32_t const expectedNodeId = expectedNodeIds[b];
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
