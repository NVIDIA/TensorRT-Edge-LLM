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

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "testUtils.h"
#include <gtest/gtest.h>

using namespace trt_edgellm;
using namespace nvinfer1;

struct SeqLensTestCase
{
    std::vector<int32_t> inputSeqLen;
    std::vector<int32_t> kvCacheStartIndices; // empty = normal prefill (all zeros)
    int32_t runtimeSeqLen;

    // Expected outputs
    std::vector<int32_t> expectedCuQSeqLens;
    std::vector<int32_t> expectedCuKVSeqLens;
    std::vector<int32_t> expectedKvCacheEndIdxs;
    std::vector<int32_t> expectedPaddedCuKVSeqLens;
};

static void verifyCalCuQCuKVSeqLensAndKVEndIdxs(SeqLensTestCase const& tc)
{
    int32_t const B = static_cast<int32_t>(tc.inputSeqLen.size());

    // Allocate GPU tensors
    rt::Tensor inputSeqLenTensor({B}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuQSeqLensTensor({B + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLensTensor({B + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor kvCacheEndIdxsTensor({B}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor paddedCuKVSeqLensTensor({B + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    CUDA_CHECK(
        cudaMemcpy(inputSeqLenTensor.rawPointer(), tc.inputSeqLen.data(), B * sizeof(int32_t), cudaMemcpyHostToDevice));

    // kvCacheStartIndices: pass empty tensor for normal prefill
    rt::Tensor kvCacheStartIdxTensor;
    if (!tc.kvCacheStartIndices.empty())
    {
        kvCacheStartIdxTensor = rt::Tensor({B}, rt::DeviceType::kGPU, DataType::kINT32);
        CUDA_CHECK(cudaMemcpy(kvCacheStartIdxTensor.rawPointer(), tc.kvCacheStartIndices.data(), B * sizeof(int32_t),
            cudaMemcpyHostToDevice));
    }

    cudaStream_t stream{nullptr};
    kernel::calCuQCuKVSeqLensAndKVEndIdxs(inputSeqLenTensor, kvCacheStartIdxTensor, cuQSeqLensTensor, cuKVSeqLensTensor,
        kvCacheEndIdxsTensor, paddedCuKVSeqLensTensor, tc.runtimeSeqLen, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Read back and verify all outputs
    auto readBack = [](rt::Tensor const& t, int32_t n) {
        std::vector<int32_t> v(n);
        CUDA_CHECK(cudaMemcpy(v.data(), t.rawPointer(), n * sizeof(int32_t), cudaMemcpyDeviceToHost));
        return v;
    };

    auto cuQ = readBack(cuQSeqLensTensor, B + 1);
    auto cuKV = readBack(cuKVSeqLensTensor, B + 1);
    auto endIdxs = readBack(kvCacheEndIdxsTensor, B);
    auto paddedCuKV = readBack(paddedCuKVSeqLensTensor, B + 1);

    for (int32_t i = 0; i <= B; ++i)
    {
        EXPECT_EQ(cuQ[i], tc.expectedCuQSeqLens[i]) << "cuQSeqLens mismatch at index " << i;
        EXPECT_EQ(cuKV[i], tc.expectedCuKVSeqLens[i]) << "cuKVSeqLens mismatch at index " << i;
        EXPECT_EQ(paddedCuKV[i], tc.expectedPaddedCuKVSeqLens[i]) << "paddedCuKVSeqLens mismatch at index " << i;
    }
    for (int32_t i = 0; i < B; ++i)
    {
        EXPECT_EQ(endIdxs[i], tc.expectedKvCacheEndIdxs[i]) << "kvCacheEndIdxs mismatch at index " << i;
    }
}

// Normal prefill, single batch
TEST(UtilKernelTest, seqLens_singleBatchNormalPrefill)
{
    verifyCalCuQCuKVSeqLensAndKVEndIdxs({
        .inputSeqLen = {128},
        .kvCacheStartIndices = {},
        .runtimeSeqLen = 128,
        .expectedCuQSeqLens = {0, 128},
        .expectedCuKVSeqLens = {0, 128},
        .expectedKvCacheEndIdxs = {128},
        .expectedPaddedCuKVSeqLens = {0, 128},
    });
}

// Normal prefill, multi-batch with different prompt lengths (the original bug scenario).
// runtimeSeqLen = max(inputSeqLen) = 128. Shorter prompt (64) is padded to 128.
TEST(UtilKernelTest, seqLens_multiBatchNormalPrefill)
{
    verifyCalCuQCuKVSeqLensAndKVEndIdxs({
        .inputSeqLen = {64, 128},
        .kvCacheStartIndices = {},
        .runtimeSeqLen = 128,
        .expectedCuQSeqLens = {0, 64, 192},
        .expectedCuKVSeqLens = {0, 64, 192},
        .expectedKvCacheEndIdxs = {128, 128},
        .expectedPaddedCuKVSeqLens = {0, 128, 256},
    });
}

// Multi-batch uniform lengths
TEST(UtilKernelTest, seqLens_multiBatchUniform)
{
    verifyCalCuQCuKVSeqLensAndKVEndIdxs({
        .inputSeqLen = {128, 128, 128},
        .kvCacheStartIndices = {},
        .runtimeSeqLen = 128,
        .expectedCuQSeqLens = {0, 128, 256, 384},
        .expectedCuKVSeqLens = {0, 128, 256, 384},
        .expectedKvCacheEndIdxs = {128, 128, 128},
        .expectedPaddedCuKVSeqLens = {0, 128, 256, 384},
    });
}

// Chunked prefill: kvCacheStartIndices > 0 for some batches
TEST(UtilKernelTest, seqLens_multiBatchChunkedPrefill)
{
    // batch 0: startIdx=0,  inputLen=128 → kvEnd=0+128=128,  actualKV=0+128=128
    // batch 1: startIdx=64, inputLen=128 → kvEnd=64+128=192, actualKV=64+128=192
    // batch 2: startIdx=32, inputLen=128 → kvEnd=32+128=160, actualKV=32+128=160
    verifyCalCuQCuKVSeqLensAndKVEndIdxs({
        .inputSeqLen = {128, 128, 128},
        .kvCacheStartIndices = {0, 64, 32},
        .runtimeSeqLen = 128,
        .expectedCuQSeqLens = {0, 128, 256, 384},
        .expectedCuKVSeqLens = {0, 128, 320, 480},
        .expectedKvCacheEndIdxs = {128, 192, 160},
        .expectedPaddedCuKVSeqLens = {0, 128, 320, 480},
    });
}

// Chunked prefill with varying input lengths
TEST(UtilKernelTest, seqLens_chunkedPrefillVaryingLengths)
{
    // batch 0: startIdx=100, inputLen=50  → kvEnd=100+50=150, actualKV=100+50=150
    // batch 1: startIdx=0,   inputLen=30  → kvEnd=0+50=50,    actualKV=0+30=30
    verifyCalCuQCuKVSeqLensAndKVEndIdxs({
        .inputSeqLen = {50, 30},
        .kvCacheStartIndices = {100, 0},
        .runtimeSeqLen = 50,
        .expectedCuQSeqLens = {0, 50, 80},
        .expectedCuKVSeqLens = {0, 150, 180},
        .expectedKvCacheEndIdxs = {150, 50},
        .expectedPaddedCuKVSeqLens = {0, 150, 200},
    });
}

// launchBuildVisionBlockRanges: image-run intervals expand per position; text and
// padding rows get the -1/-1 sentinel (empty interval).
TEST(UtilKernelTest, visionBlockRanges_runsSentinelsAndPadding)
{
    int32_t constexpr batchSize = 2;
    int32_t constexpr seqLen = 12;
    // Batch 0: block 0 = [1, 5], block 1 = [7, 9]; full context.
    // Batch 1: block 0 = [0, 1]; block 1 = [4, 8] but contextLength = 7 clips
    //          the run to [4, 6]; positions >= 7 are padding.
    std::vector<int32_t> const blockIds{-1, 0, 0, 0, 0, 0, -1, 1, 1, 1, -1, -1, //
        0, 0, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1};
    std::vector<int32_t> const contextLengths{seqLen, 7};
    std::vector<int32_t> const expectedBegin{-1, 1, 1, 1, 1, 1, -1, 7, 7, 7, -1, -1, //
        0, 0, -1, -1, 4, 4, 4, -1, -1, -1, -1, -1};
    std::vector<int32_t> const expectedEnd{-1, 5, 5, 5, 5, 5, -1, 9, 9, 9, -1, -1, //
        1, 1, -1, -1, 6, 6, 6, -1, -1, -1, -1, -1};

    rt::Tensor idsTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor lengthsTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor beginTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor endTensor({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(idsTensor, blockIds);
    copyHostToDevice(lengthsTensor, contextLengths);

    cudaStream_t stream{nullptr};
    kernel::launchBuildVisionBlockRanges(idsTensor.dataPointer<int32_t>(), lengthsTensor.dataPointer<int32_t>(),
        beginTensor.dataPointer<int32_t>(), endTensor.dataPointer<int32_t>(), batchSize, seqLen, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto const begin = copyDeviceToHost<int32_t>(beginTensor);
    auto const end = copyDeviceToHost<int32_t>(endTensor);
    for (size_t i = 0; i < expectedBegin.size(); ++i)
    {
        EXPECT_EQ(begin[i], expectedBegin[i]) << "blockBegin mismatch at flat index " << i;
        EXPECT_EQ(end[i], expectedEnd[i]) << "blockEnd mismatch at flat index " << i;
    }
}
