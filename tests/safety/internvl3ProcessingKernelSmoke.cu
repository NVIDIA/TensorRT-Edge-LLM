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

#include "safetyTestLifecycle.h"

#include "safety_inference/embedding/fp16Embedding.h"
#include "safety_inference/image_preprocess/models/internvl3/internvl3PreprocessKernels.cuh"
#include "safety_inference/image_preprocess/rgbResizeNormalizeKernels.cuh"
#include "safety_inference/sampler/greedyArgmaxKernel.cuh"

#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
#include <cuda_runtime_api_safe_ex.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace
{
using namespace trt_edgellm::safety;
using trt_edgellm::safety::engine_test::TestLifecycle;
using trt_edgellm::safety::engine_test::finishTestProcess;

bool check(cudaError_t status, char const* operation)
{
    if (status == cudaSuccess)
    {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool expectError(cudaError_t status, cudaError_t expected, char const* operation)
{
    if (status == expected)
    {
        return true;
    }
    std::fprintf(
        stderr, "%s returned %s instead of %s\n", operation, cudaGetErrorString(status), cudaGetErrorString(expected));
    return false;
}

bool prepareForOperational()
{
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
    return check(cudaSafeExNotifyNvDvmsTransitionToOperational(), "cudaSafeExNotifyNvDvmsTransitionToOperational");
#else
    return true;
#endif
}

template <typename T>
struct Buffer
{
    T* host{nullptr};
    T* device{nullptr};
    std::size_t count{0};

    bool allocate(std::size_t elements)
    {
        count = elements;
        return check(cudaMallocHost(reinterpret_cast<void**>(&host), count * sizeof(T)), "cudaMallocHost")
            && check(cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(T)), "cudaMalloc");
    }

    bool upload(cudaStream_t stream) const
    {
        return check(cudaMemcpyAsync(device, host, count * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
    }

    bool download(cudaStream_t stream) const
    {
        return check(cudaMemcpyAsync(host, device, count * sizeof(T), cudaMemcpyDeviceToHost, stream), "download");
    }

    bool release()
    {
        bool const deviceReleased = device == nullptr || check(cudaFree(device), "cudaFree");
        bool const hostReleased = host == nullptr || check(cudaFreeHost(host), "cudaFreeHost");
        device = nullptr;
        host = nullptr;
        return deviceReleased && hostReleased;
    }
};

bool testEmbedding(cudaStream_t stream)
{
    Buffer<int32_t> tokens{};
    Buffer<half> table{};
    Buffer<half> output{};
    if (!tokens.allocate(2) || !table.allocate(12) || !output.allocate(8))
    {
        return false;
    }
    tokens.host[0] = 2;
    tokens.host[1] = 0;
    for (int32_t index = 0; index < 12; ++index)
    {
        table.host[index] = __float2half(static_cast<float>(index));
    }
    bool success = tokens.upload(stream) && table.upload(stream);
    // Given invalid and valid lookup arguments, when the production launcher runs, then it rejects invalid input and
    // copies each selected FP16 row.
    success = success
        && expectError(launchFp16EmbeddingLookup(nullptr, 2, table.device, 3, 4, output.device, stream),
            cudaErrorInvalidValue, "invalid FP16 embedding lookup")
        && check(launchFp16EmbeddingLookup(tokens.device, 2, table.device, 3, 4, output.device, stream),
            "FP16 embedding lookup")
        && output.download(stream) && check(cudaStreamSynchronize(stream), "embedding synchronize");
    for (int32_t index = 0; success && index < 4; ++index)
    {
        success = __half2float(output.host[index]) == static_cast<float>(8 + index)
            && __half2float(output.host[4 + index]) == static_cast<float>(index);
    }
    return output.release() && table.release() && tokens.release() && success;
}

bool testSampler(cudaStream_t stream)
{
    Buffer<float> logits{};
    Buffer<ArgmaxPair> result{};
    if (!logits.allocate(300) || !result.allocate(1))
    {
        return false;
    }
    std::fill_n(logits.host, logits.count, std::numeric_limits<float>::quiet_NaN());
    logits.host[9] = 4.0F;
    logits.host[257] = 4.0F;
    bool success = logits.upload(stream);
    // Given invalid arguments and tied maxima across reduction warps, when the production launcher runs, then it
    // rejects invalid input and selects the lower token ID.
    success = success
        && expectError(
            launchGreedyArgmax(logits.device, 0, result.device, stream), cudaErrorInvalidValue, "invalid greedy argmax")
        && check(launchGreedyArgmax(logits.device, 300, result.device, stream), "greedy argmax")
        && result.download(stream) && check(cudaStreamSynchronize(stream), "sampler synchronize")
        && result.host[0].valid == 1 && result.host[0].index == 9 && result.host[0].value == 4.0F;
    std::fill_n(logits.host, logits.count, std::numeric_limits<float>::quiet_NaN());
    // Given only unordered values, when greedy argmax runs, then it reports that no token is valid.
    success = success && logits.upload(stream)
        && check(launchGreedyArgmax(logits.device, 300, result.device, stream), "unordered greedy argmax")
        && result.download(stream) && check(cudaStreamSynchronize(stream), "unordered sampler synchronize")
        && result.host[0].valid == 0 && result.host[0].index == -1;
    return result.release() && logits.release() && success;
}

bool close(float actual, float expected)
{
    return std::isfinite(actual) && std::abs(actual - expected) < 0.003F;
}

bool validateTiles(half const* output, int32_t tileCount, float const (&expected)[kInternVLChannels])
{
    for (int32_t tile = 0; tile < tileCount; ++tile)
    {
        for (int32_t channel = 0; channel < kInternVLChannels; ++channel)
        {
            std::size_t const first
                = static_cast<std::size_t>(tile) * kInternVLTileValues + channel * kInternVLTilePixels;
            if (!close(__half2float(output[first]), expected[channel])
                || !close(__half2float(output[first + kInternVLTilePixels - 1]), expected[channel]))
            {
                return false;
            }
        }
    }
    return true;
}

bool testImage(cudaStream_t stream)
{
    Buffer<uint8_t> source{};
    Buffer<half> canvas{};
    Buffer<half> output{};
    InternVLTilePlan plan{};
    if (!computeInternVLTilePlan(896, 448, 2, 2, true, plan) || plan.mainTiles != 2 || !plan.thumbnail
        || plan.totalTiles != 3 || !source.allocate(3) || !canvas.allocate(3 * kInternVLTileValues)
        || !output.allocate(3 * kInternVLTileValues))
    {
        return false;
    }
    source.host[0] = 255;
    source.host[1] = 0;
    source.host[2] = 127;
    bool success = source.upload(stream);
    RgbNormalization const normalization{1.0F / 255.0F, {0.485F, 0.456F, 0.406F}, {0.229F, 0.224F, 0.225F}, 0.5F};
    RgbNormalization invalidNormalization = normalization;
    invalidNormalization.deviation[1] = 0.0F;
    InternVLTilePlan invalidPlan = plan;
    ++invalidPlan.totalTiles;
    // Given malformed plans and normalization, when preprocessing is launched, then validation rejects them.
    success = success
        && expectError(launchInternVLPreprocess(source.device, 1, 1, kInternVLChannels, invalidPlan, normalization,
                           canvas.device, output.device, stream),
            cudaErrorInvalidValue, "invalid InternVL tile plan")
        && expectError(launchInternVLPreprocess(source.device, 1, 1, kInternVLChannels, plan, invalidNormalization,
                           canvas.device, output.device, stream),
            cudaErrorInvalidValue, "invalid InternVL normalization");
    // Given a constant RGB image and a thumbnail plan, when production preprocessing runs, then the resized main
    // tiles and thumbnail retain channel normalization and NCHW placement.
    success = success
        && check(cudaMemsetAsync(output.device, 0, output.count * sizeof(half), stream), "clear image output")
        && check(launchInternVLPreprocess(
                     source.device, 1, 1, kInternVLChannels, plan, normalization, canvas.device, output.device, stream),
            "InternVL preprocessing")
        && output.download(stream) && check(cudaStreamSynchronize(stream), "image synchronize");
    float const expected[kInternVLChannels]{(1.0F - normalization.mean[0]) / normalization.deviation[0],
        (0.0F - normalization.mean[1]) / normalization.deviation[1],
        (127.0F * normalization.scale - normalization.mean[2]) / normalization.deviation[2]};
    success = success && validateTiles(output.host, plan.totalTiles, expected);
    return output.release() && canvas.release() && source.release() && success;
}

bool smoke()
{
    TestLifecycle lifecycle{};
    cudaStream_t stream{nullptr};
    if (!lifecycle.initialize()
        || !check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"))
    {
        return false;
    }
    bool success = testEmbedding(stream) && testSampler(stream) && testImage(stream);
    success = success && prepareForOperational() && lifecycle.operational();
    success = success && testEmbedding(stream) && testSampler(stream) && testImage(stream);
    success = success && lifecycle.deinitialize();
    success = check(cudaStreamDestroy(stream), "cudaStreamDestroy") && success;
    return lifecycle.finish() && success;
}
} // namespace

int main(int argc, char** argv)
{
#if defined(__QNXNTO__) && !defined(EDGELLM_TEST_QNX_STANDARD)
    if (argc != 2 || std::strcmp(argv[1], "--external-lifecycle") != 0)
    {
        std::fprintf(stderr, "QNX Safety requires --external-lifecycle and an exclusive VM lifecycle coordinator.\n");
        return finishTestProcess(2);
    }
#else
    if (argc != 1)
    {
        std::fprintf(stderr, "Usage: %s\n", argv[0]);
        return 2;
    }
#endif
    bool const success = smoke();
    std::printf("%s InternVL3 processing kernels\n", success ? "PASS" : "FAIL");
    return finishTestProcess(success ? 0 : 1);
}
