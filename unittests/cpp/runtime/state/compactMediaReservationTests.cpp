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

#include "runtime/state/compactMediaReservation.h"

#include "common/cudaMacros.h"
#include "kernels/embeddingKernels/embeddingKernels.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace trt_edgellm::rt
{

// Given slot 1 of a reservation with separate image (E = 4) and audio (E = 6) rows and one deepstack layer, and a
//       prompt holding a two-token image inside a reused context-cache prefix, then interleaved image and audio tokens,
//       with encoder outputs and deepstack features covering every media position
// When the prompt is staged from that prefill start, its slices are mapped, and the deepstack assembly gathers the
//      slice after the prefix
// Then image positions map into the image index space (1 * 4 + i) and audio positions into the audio one (1 * 6 + i),
//      each counting only the rows after the prefix, which map to no row; the stored rows are those after the prefix;
//      and the deepstack gather gives each image token its own feature row and every other token zeros
TEST(CompactMediaReservationTest, StagesOnlyRowsAfterThePrefixInPerKindIndexSpaces)
{
    constexpr int32_t kHidden{16};
    constexpr int32_t kImage{90};
    constexpr int32_t kAudio{91};
    constexpr int32_t kText{5};
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Every (source, row, channel) holds a distinct exact FP16 value; source 0 stands for a zero row.
    auto const value = [](int32_t source, int64_t row, int64_t channel) {
        return source == 0 ? 0.0F : static_cast<float>(source * 256 + row * kHidden + channel + 1);
    };
    auto const makeSource = [&](int32_t source, int64_t rows) {
        Tensor tensor({rows, kHidden}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
        std::vector<half> host;
        for (int64_t index = 0; index < rows * kHidden; ++index)
        {
            host.push_back(__float2half(value(source, index / kHidden, index % kHidden)));
        }
        CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice));
        return tensor;
    };
    // Rows from first on of a [rows, kHidden] FP16 device tensor against value() of each (source, row).
    auto const expectRows = [&](Tensor const& tensor, int64_t first, std::vector<std::pair<int32_t, int64_t>> rows) {
        std::vector<half> host(rows.size() * kHidden);
        CUDA_CHECK(cudaMemcpy(host.data(), static_cast<half const*>(tensor.rawPointer()) + first * kHidden,
            host.size() * sizeof(half), cudaMemcpyDeviceToHost));
        for (size_t index = 0; index < host.size(); ++index)
        {
            auto const [source, row] = rows[index / kHidden];
            EXPECT_EQ(__half2float(host[index]), value(source, row, index % kHidden)) << "row " << index / kHidden;
        }
    };
    CompactMediaReservation reservation({/*slots=*/2, /*imageRowsPerSlot=*/4, /*audioRowsPerSlot=*/6, kHidden, 1});
    Tensor const image = makeSource(1, 4);
    Tensor const audio = makeSource(2, 3);
    Tensor const feature = makeSource(3, 4);
    std::vector<int32_t> const prompt{kImage, kImage, kText, kAudio, kImage, kText, kAudio, kImage, kAudio};
    reservation.stage(1, prompt, /*prefillStart=*/3, kImage, kAudio, {&image, {std::cref(feature)}, &audio}, stream);

    std::vector<int32_t> prefix(3, 7777);
    reservation.mapSlice(1, 0, 3, prefix.data());
    EXPECT_EQ(prefix, (std::vector<int32_t>{-1, -1, -1}));
    std::vector<int32_t> slice(6, 7777);
    reservation.mapSlice(1, 3, 6, slice.data());
    EXPECT_EQ(slice, (std::vector<int32_t>{6, 4, -1, 7, 5, 8}));
    expectRows(reservation.imageRows(), 4, {{1, 2}, {1, 3}});
    expectRows(reservation.audioRows(), 6, {{2, 0}, {2, 1}, {2, 2}});

    Tensor tokenIds({1, 6}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor indices({1, 6}, DeviceType::kGPU, trt_edgellm::DataType::kINT32);
    Tensor output({1, 6, kHidden}, DeviceType::kGPU, trt_edgellm::DataType::kHALF);
    CUDA_CHECK(cudaMemcpy(tokenIds.rawPointer(), prompt.data() + 3, 6 * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(indices.rawPointer(), slice.data(), 6 * sizeof(int32_t), cudaMemcpyHostToDevice));
    kernel::assembleDeepstackEmbedding(
        tokenIds, reservation.deepstackRows().front().get(), output, stream, kImage, std::cref(indices));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    expectRows(output, 0, {{0, 0}, {3, 2}, {0, 0}, {0, 0}, {3, 3}, {0, 0}});
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace trt_edgellm::rt
