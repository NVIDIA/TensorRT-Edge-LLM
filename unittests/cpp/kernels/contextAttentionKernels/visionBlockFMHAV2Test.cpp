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

#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/cuteDslFMHAV2Runner.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;

namespace
{

TEST(VisionBlockRangeBuilderTest, ExpandsContiguousBlocksAndMasksPadding)
{
    int32_t constexpr batchSize = 2;
    int32_t constexpr seqLen = 8;
    std::vector<int32_t> const blockIds{
        -1,
        0,
        0,
        0,
        -1,
        1,
        1,
        -1,
        2,
        2,
        -1,
        3,
        3,
        3,
        3,
        4,
    };
    std::vector<int32_t> const contextLengths{7, 6};
    std::vector<int32_t> const expectedBegin{
        -1,
        1,
        1,
        1,
        -1,
        5,
        5,
        -1,
        0,
        0,
        -1,
        3,
        3,
        3,
        -1,
        -1,
    };
    std::vector<int32_t> const expectedEnd{
        -1,
        3,
        3,
        3,
        -1,
        6,
        6,
        -1,
        1,
        1,
        -1,
        5,
        5,
        5,
        -1,
        -1,
    };

    rt::Tensor blockIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor contextLengthsTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor blockBeginTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor blockEndTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(blockIdsTensor, blockIds);
    copyHostToDevice(contextLengthsTensor, contextLengths);

    kernel::launchBuildVisionBlockRanges(blockIdsTensor.dataPointer<int32_t>(),
        contextLengthsTensor.dataPointer<int32_t>(), blockBeginTensor.dataPointer<int32_t>(),
        blockEndTensor.dataPointer<int32_t>(), batchSize, seqLen, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_EQ(copyDeviceToHost<int32_t>(blockBeginTensor), expectedBegin);
    EXPECT_EQ(copyDeviceToHost<int32_t>(blockEndTensor), expectedEnd);
}

std::vector<half> visionBlockReference(std::vector<half> const& q, std::vector<half> const& k,
    std::vector<half> const& v, std::vector<int32_t> const& blockIds, int32_t seqLen, int32_t numQHeads,
    int32_t numKVHeads, int32_t headDim, int32_t slidingWindow)
{
    std::vector<half> output(q.size(), __float2half(0.0F));
    int32_t const groupSize = numQHeads / numKVHeads;
    float const scale = 1.0F / std::sqrt(static_cast<float>(headDim));
    std::vector<float> logits(seqLen);
    std::vector<float> accumulator(headDim);

    auto offset = [=](int32_t token, int32_t head, int32_t heads) {
        return (static_cast<size_t>(token) * heads + head) * headDim;
    };

    for (int32_t query = 0; query < seqLen; ++query)
    {
        for (int32_t qHead = 0; qHead < numQHeads; ++qHead)
        {
            int32_t const kvHead = qHead / groupSize;
            half const* qRow = q.data() + offset(query, qHead, numQHeads);
            float maxLogit = -INFINITY;
            std::fill(logits.begin(), logits.end(), -INFINITY);
            for (int32_t key = 0; key < seqLen; ++key)
            {
                bool const localCausal = key <= query && key > query - slidingWindow;
                bool const sameBlock = blockIds[query] >= 0 && blockIds[query] == blockIds[key];
                if (!localCausal && !sameBlock)
                {
                    continue;
                }
                half const* kRow = k.data() + offset(key, kvHead, numKVHeads);
                float dot = 0.0F;
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    dot += __half2float(qRow[dim]) * __half2float(kRow[dim]);
                }
                logits[key] = dot * scale;
                maxLogit = std::max(maxLogit, logits[key]);
            }

            float denominator = 0.0F;
            std::fill(accumulator.begin(), accumulator.end(), 0.0F);
            for (int32_t key = 0; key < seqLen; ++key)
            {
                float const weight = std::isfinite(logits[key]) ? std::exp(logits[key] - maxLogit) : 0.0F;
                denominator += weight;
                half const* vRow = v.data() + offset(key, kvHead, numKVHeads);
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    accumulator[dim] += weight * __half2float(vRow[dim]);
                }
            }
            half* outRow = output.data() + offset(query, qHead, numQHeads);
            for (int32_t dim = 0; dim < headDim; ++dim)
            {
                outRow[dim] = __float2half(accumulator[dim] / denominator);
            }
        }
    }
    return output;
}

//! @param slidingWindow Left window in the sliding_window_size convention (counts the query itself).
//! Pass a value >= seqLen to exercise the full-causal Gemma4 global-layer configuration, which
//! reaches the kernel as an unbounded runtime window.
void RunVisionBlockCase(int32_t headDim, int32_t numQHeads, int32_t numKVHeads, int32_t slidingWindow)
{
    int32_t constexpr batchSize = 1;
    int32_t constexpr seqLen = 64;
    bool const unboundedWindow = slidingWindow >= seqLen;

    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!CuteDslFMHAV2Runner::canImplement(
            numQHeads, numKVHeads, headDim, smVersion, DataType::kHALF, CuteDslFMHAV2MaskType::kVISION_BLOCK))
    {
        GTEST_SKIP() << "FMHA-v2 CuTe DSL vision-block kernel is unavailable on SM" << smVersion;
    }

    std::vector<int32_t> blockIds(seqLen, -1);
    std::fill(blockIds.begin() + 8, blockIds.begin() + 24, 0);
    std::fill(blockIds.begin() + 40, blockIds.begin() + 56, 1);
    std::vector<half> q(static_cast<size_t>(seqLen) * numQHeads * headDim);
    std::vector<half> k(static_cast<size_t>(seqLen) * numKVHeads * headDim);
    std::vector<half> v(k.size());
    uniformFloatInitialization(q, -0.25F, 0.25F);
    uniformFloatInitialization(k, -0.25F, 0.25F);
    uniformFloatInitialization(v, -0.25F, 0.25F);
    auto const expected
        = visionBlockReference(q, k, v, blockIds, seqLen, numQHeads, numKVHeads, headDim, slidingWindow);

    rt::Tensor qTensor({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kTensor({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputTensor({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor blockIdsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor contextLengthsTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor blockBeginTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor blockEndTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(qTensor, q);
    copyHostToDevice(kTensor, k);
    copyHostToDevice(vTensor, v);
    copyHostToDevice(blockIdsTensor, blockIds);
    copyHostToDevice(contextLengthsTensor, std::vector<int32_t>{seqLen});
    copyHostToDevice(cuSeqLensTensor, std::vector<int32_t>{0, seqLen});

    cudaStream_t stream{nullptr};
    kernel::launchBuildVisionBlockRanges(blockIdsTensor.dataPointer<int32_t>(),
        contextLengthsTensor.dataPointer<int32_t>(), blockBeginTensor.dataPointer<int32_t>(),
        blockEndTensor.dataPointer<int32_t>(), batchSize, seqLen, stream);
    CUDA_CHECK(cudaFree(nullptr));
    CuteDslFMHAV2Runner runner(numQHeads, numKVHeads, headDim, batchSize, seqLen, seqLen);
    ASSERT_TRUE(runner.runVisionBlock(qTensor.rawPointer(), kTensor.rawPointer(), vTensor.rawPointer(),
        outputTensor.rawPointer(), cuSeqLensTensor.dataPointer<int32_t>(), blockBeginTensor.dataPointer<int32_t>(),
        blockEndTensor.dataPointer<int32_t>(), stream, 1.0F / std::sqrt(static_cast<float>(headDim)),
        unboundedWindow ? -1 : slidingWindow - 1));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const actual = copyDeviceToHost<half>(outputTensor);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
    {
        ASSERT_TRUE(isclose(actual[i], expected[i], 2e-2F, 2e-2F))
            << "FMHA-v2 vision-block mismatch at D=" << headDim << " window=" << slidingWindow << " index " << i
            << ": actual=" << __half2float(actual[i]) << " expected=" << __half2float(expected[i]);
    }
}

TEST(VisionBlockFMHAV2Test, MatchesReference)
{
    RunVisionBlockCase(/*headDim=*/256, /*numQHeads=*/4, /*numKVHeads=*/2, /*slidingWindow=*/16);
    RunVisionBlockCase(/*headDim=*/256, /*numQHeads=*/4, /*numKVHeads=*/2, /*slidingWindow=*/64);
}

// Gemma4 Unified global layers: D512, Hq=16, Hkv=1, full-causal (no window).
TEST(VisionBlockFMHAV2Test, MatchesReferenceD512)
{
    RunVisionBlockCase(/*headDim=*/512, /*numQHeads=*/16, /*numKVHeads=*/1, /*slidingWindow=*/16);
    RunVisionBlockCase(/*headDim=*/512, /*numQHeads=*/16, /*numKVHeads=*/1, /*slidingWindow=*/64);
}

} // namespace
