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

#include "common/tensor.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "testUtils.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

int32_t constexpr kImageTok = 50;
int32_t constexpr kAudioTok = 99;
int32_t constexpr kTextTok = 10;

// Create a GPU INT32 tensor from a flat host vector with shape [batchSize, seqLen].
rt::Tensor makeGpuIds(std::vector<int32_t> const& ids, int64_t batchSize, int64_t seqLen)
{
    rt::Tensor host({batchSize, seqLen}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(host.rawPointer(), ids.data(), ids.size() * sizeof(int32_t));
    rt::Tensor device({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    cudaMemcpy(device.rawPointer(), host.rawPointer(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    return device;
}

// Create a GPU INT32 tensor from a host vector with shape [n].
rt::Tensor makeGpuOffsets(std::vector<int32_t> const& offsets)
{
    rt::Tensor device({static_cast<int64_t>(offsets.size())}, rt::DeviceType::kGPU, DataType::kINT32);
    cudaMemcpy(device.rawPointer(), offsets.data(), offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    return device;
}

// Read GPU result back to host vector.
std::vector<int32_t> readBack(rt::Tensor const& t)
{
    auto const shape = t.getShape();
    int64_t const n = shape.volume();
    std::vector<int32_t> v(n);
    cudaMemcpy(v.data(), t.rawPointer(), n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    return v;
}

} // namespace

// Basic test: no offsets, single batch — verify CUDA kernel matches host reference.
TEST(GenerateMultimodalIndicesGpu, BasicNoOffset)
{
    // batch=1, seqLen=6: [text, img, img, text, img, text]
    auto ids = makeGpuIds({kTextTok, kImageTok, kImageTok, kTextTok, kImageTok, kTextTok}, 1, 6);
    rt::Tensor output({1, 6}, rt::DeviceType::kGPU, DataType::kINT32);

    kernel::generateMultimodalIndices(ids, output, kImageTok, std::nullopt, nullptr);
    cudaDeviceSynchronize();

    auto result = readBack(output);
    EXPECT_EQ(result, (std::vector<int32_t>{0, 0, 1, 0, 2, 0}));
}

// Single batch with image offset: simulates prefix having 3 image tokens reused.
// Suffix has 2 image tokens that should get indices 3 and 4.
TEST(GenerateMultimodalIndicesGpu, SingleBatchImageOffset)
{
    // Suffix tokens: [text, img, text, img, text]
    // Prefix had 3 image tokens (reused, not in suffix)
    auto ids = makeGpuIds({kTextTok, kImageTok, kTextTok, kImageTok, kTextTok}, 1, 5);
    rt::Tensor output({1, 5}, rt::DeviceType::kGPU, DataType::kINT32);

    std::vector<int32_t> imageOffsets = {3}; // prefix had 3 image tokens
    auto offsetsGpu = makeGpuOffsets(imageOffsets);

    kernel::generateMultimodalIndices(
        ids, output, kImageTok, std::nullopt, nullptr, offsetsGpu.dataPointer<int32_t>(), nullptr);
    cudaDeviceSynchronize();

    auto result = readBack(output);
    // Image tokens should get indices 3 and 4 (not 0 and 1)
    EXPECT_EQ(result, (std::vector<int32_t>{0, 3, 0, 4, 0}));
}

// Single batch with audio offset.
TEST(GenerateMultimodalIndicesGpu, SingleBatchAudioOffset)
{
    // Suffix tokens: [audio, text, audio]
    // Prefix had 5 audio tokens
    auto ids = makeGpuIds({kAudioTok, kTextTok, kAudioTok}, 1, 3);
    rt::Tensor output({1, 3}, rt::DeviceType::kGPU, DataType::kINT32);

    std::vector<int32_t> audioOffsets = {5};
    auto offsetsGpu = makeGpuOffsets(audioOffsets);

    kernel::generateMultimodalIndices(
        ids, output, std::nullopt, kAudioTok, nullptr, nullptr, offsetsGpu.dataPointer<int32_t>());
    cudaDeviceSynchronize();

    auto result = readBack(output);
    EXPECT_EQ(result, (std::vector<int32_t>{5, 0, 6}));
}

// Multi-image KV reuse: img0 (3 tokens) entirely in prefix, img1 (2 tokens) in suffix.
// Embedding tensor: [img0_rows(3) | img1_rows(2)]. Suffix img1 should get indices 3, 4.
TEST(GenerateMultimodalIndicesGpu, MultiImagePrefixReuse)
{
    // Full sequence was: [img0, img0, img0, text, text, img1, img1, text]
    // Prefix reuse covers positions 0-4 (img0 + text), suffix is positions 5-7: [img1, img1, text]
    auto ids = makeGpuIds({kImageTok, kImageTok, kTextTok}, 1, 3);
    rt::Tensor output({1, 3}, rt::DeviceType::kGPU, DataType::kINT32);

    // Prefix had 3 image tokens (img0)
    std::vector<int32_t> imageOffsets = {3};
    auto offsetsGpu = makeGpuOffsets(imageOffsets);

    kernel::generateMultimodalIndices(
        ids, output, kImageTok, std::nullopt, nullptr, offsetsGpu.dataPointer<int32_t>(), nullptr);
    cudaDeviceSynchronize();

    auto result = readBack(output);
    // img1 tokens get indices 3, 4 (skipping img0's 3 rows)
    EXPECT_EQ(result, (std::vector<int32_t>{3, 4, 0}));
}

// Multi-batch with different prefix reuse per batch.
// Batch 0: prefix had img0 (4 tokens), suffix has img1 (2 tokens)
// Batch 1: no prefix reuse, suffix has img2 (3 tokens)
// Embedding tensor: [img0(4) | img1(2) | img2(3)]
// Batch 0 offset = cumulative(0) + prefix(4) = 4
// Batch 1 offset = cumulative(4+2=6) + prefix(0) = 6
TEST(GenerateMultimodalIndicesGpu, MultiBatchDifferentOffsets)
{
    // batch=2, seqLen=4
    // Batch 0 suffix: [img, img, text, text] (img1, 2 tokens)
    // Batch 1 suffix: [img, img, img, text]  (img2, 3 tokens)
    auto ids = makeGpuIds({kImageTok, kImageTok, kTextTok, kTextTok, kImageTok, kImageTok, kImageTok, kTextTok}, 2, 4);
    rt::Tensor output({2, 4}, rt::DeviceType::kGPU, DataType::kINT32);

    // Batch 0: cumulative prior total = 0, prefix img count = 4 → offset = 4
    // Batch 1: cumulative prior total = 4+2=6, prefix img count = 0 → offset = 6
    std::vector<int32_t> imageOffsets = {4, 6};
    auto offsetsGpu = makeGpuOffsets(imageOffsets);

    kernel::generateMultimodalIndices(
        ids, output, kImageTok, std::nullopt, nullptr, offsetsGpu.dataPointer<int32_t>(), nullptr);
    cudaDeviceSynchronize();

    auto result = readBack(output);
    // Batch 0: img tokens get 4, 5
    // Batch 1: img tokens get 6, 7, 8
    EXPECT_EQ(result, (std::vector<int32_t>{4, 5, 0, 0, 6, 7, 8, 0}));
}

// Multi-batch with both image and audio offsets.
TEST(GenerateMultimodalIndicesGpu, MultiBatchMixedMediaOffsets)
{
    // batch=2, seqLen=4
    // Batch 0 suffix: [img, audio, text, text]
    // Batch 1 suffix: [audio, img, img, text]
    auto ids = makeGpuIds({kImageTok, kAudioTok, kTextTok, kTextTok, kAudioTok, kImageTok, kImageTok, kTextTok}, 2, 4);
    rt::Tensor output({2, 4}, rt::DeviceType::kGPU, DataType::kINT32);

    // Image offsets: batch0=2 (prefix had 2 img tokens), batch1=2+1=3 (cumulative 2+1, prefix 0)
    // Audio offsets: batch0=1 (prefix had 1 audio token), batch1=1+1=2 (cumulative 1+1, prefix 0)
    std::vector<int32_t> imageOffsets = {2, 3};
    std::vector<int32_t> audioOffsets = {1, 2};
    auto imgOffsetsGpu = makeGpuOffsets(imageOffsets);
    auto audOffsetsGpu = makeGpuOffsets(audioOffsets);

    kernel::generateMultimodalIndices(ids, output, kImageTok, kAudioTok, nullptr, imgOffsetsGpu.dataPointer<int32_t>(),
        audOffsetsGpu.dataPointer<int32_t>());
    cudaDeviceSynchronize();

    auto result = readBack(output);
    // Batch 0: img→2, audio→1
    // Batch 1: audio→2, img→3, img→4
    EXPECT_EQ(result, (std::vector<int32_t>{2, 1, 0, 0, 2, 3, 4, 0}));
}

// Null offsets (no prefix reuse) should behave identically to the original kernel.
TEST(GenerateMultimodalIndicesGpu, NullOffsetsMatchesOriginal)
{
    // batch=2, seqLen=3
    auto ids = makeGpuIds({kImageTok, kTextTok, kImageTok, kImageTok, kTextTok, kImageTok}, 2, 3);
    rt::Tensor output({2, 3}, rt::DeviceType::kGPU, DataType::kINT32);

    kernel::generateMultimodalIndices(ids, output, kImageTok, std::nullopt, nullptr, nullptr, nullptr);
    cudaDeviceSynchronize();

    auto result = readBack(output);
    // Global counter: batch0 gets 0, 1; batch1 continues at 2, 3
    EXPECT_EQ(result, (std::vector<int32_t>{0, 0, 1, 2, 0, 3}));
}

// Zero offsets should behave the same as null offsets for batch size 1.
TEST(GenerateMultimodalIndicesGpu, ZeroOffsetsSingleBatch)
{
    auto ids = makeGpuIds({kImageTok, kTextTok, kImageTok}, 1, 3);
    rt::Tensor output({1, 3}, rt::DeviceType::kGPU, DataType::kINT32);

    std::vector<int32_t> imageOffsets = {0};
    auto offsetsGpu = makeGpuOffsets(imageOffsets);

    kernel::generateMultimodalIndices(
        ids, output, kImageTok, std::nullopt, nullptr, offsetsGpu.dataPointer<int32_t>(), nullptr);
    cudaDeviceSynchronize();

    auto result = readBack(output);
    EXPECT_EQ(result, (std::vector<int32_t>{0, 0, 1}));
}
