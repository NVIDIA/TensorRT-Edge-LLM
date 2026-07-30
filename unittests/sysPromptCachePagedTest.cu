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

// System-prompt cache capture/restore paged-pool round-trip. captureKVCache must read a sequence
// out of row `b` of the NHD pool [2, maxBatch, capPadded, H, D] (K from the K-half, V from the
// V-half) into a saved tensor [2, seqLen, H, D]; restoreKVCache must write it back to the same NHD
// positions. We assert the NHD addressing explicitly: a per-token, per-(head,dim) value pattern
// survives the round-trip in BOTH halves, and tokens beyond the captured prefix are left untouched.

#include "common/cudaUtils.h"
#include "runtime/hybridCacheManager.h"
#include "testUtils.h"
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

// Distinct value per (half, layer, token, headDim-flattened-elem). Encoded so any
// mis-addressing (wrong token stride, wrong head, K/V swap) produces a mismatch.
float patternVal(int32_t isV, int32_t L, int32_t token, int32_t elem)
{
    return static_cast<float>(isV * 100000 + (L + 1) * 10000 + (token + 1) * 100 + elem) / 7.0f;
}

// Write the value pattern into the first `seqLen` tokens of row `b` of a half-view
// [maxBatch, capPadded, H, D]; leave the rest of the row at a sentinel.
void seedHalf(rt::Tensor& halfView, int32_t b, int32_t isV, int32_t L, int32_t seqLen, float sentinel)
{
    auto const& shape = halfView.getShape();
    int32_t const capPadded = static_cast<int32_t>(shape[1]);
    int32_t const headsTimesDim = static_cast<int32_t>(shape[2] * shape[3]);
    int64_t const rowElems = static_cast<int64_t>(capPadded) * headsTimesDim;

    std::vector<half> host(static_cast<size_t>(rowElems), __float2half(sentinel));
    for (int32_t t = 0; t < seqLen; ++t)
    {
        for (int32_t e = 0; e < headsTimesDim; ++e)
        {
            host[static_cast<size_t>(static_cast<int64_t>(t) * headsTimesDim + e)]
                = __float2half(patternVal(isV, L, t, e));
        }
    }
    int64_t const off = static_cast<int64_t>(b) * rowElems;
    CUDA_CHECK(cudaMemcpy(static_cast<half*>(halfView.rawPointer()) + off, host.data(),
        static_cast<size_t>(rowElems) * sizeof(half), cudaMemcpyHostToDevice));
}

// Overwrite the whole row of a half-view with `value`.
void fillRow(rt::Tensor& halfView, int32_t b, float value)
{
    auto const& shape = halfView.getShape();
    int64_t rowElems = 1;
    for (int32_t d = 1; d < shape.getNumDims(); ++d)
    {
        rowElems *= shape[d];
    }
    std::vector<half> host(static_cast<size_t>(rowElems), __float2half(value));
    int64_t const off = static_cast<int64_t>(b) * rowElems;
    CUDA_CHECK(cudaMemcpy(static_cast<half*>(halfView.rawPointer()) + off, host.data(),
        static_cast<size_t>(rowElems) * sizeof(half), cudaMemcpyHostToDevice));
}

std::vector<half> readRow(rt::Tensor const& halfView, int32_t b)
{
    auto const& shape = halfView.getShape();
    int64_t rowElems = 1;
    for (int32_t d = 1; d < shape.getNumDims(); ++d)
    {
        rowElems *= shape[d];
    }
    std::vector<half> host(static_cast<size_t>(rowElems));
    int64_t const off = static_cast<int64_t>(b) * rowElems;
    CUDA_CHECK(cudaMemcpy(host.data(), static_cast<half const*>(halfView.rawPointer()) + off,
        static_cast<size_t>(rowElems) * sizeof(half), cudaMemcpyDeviceToHost));
    return host;
}

rt::HybridCacheManager::Config makeUniformAttnConfig(
    int32_t numLayers, int32_t maxBatch, int32_t maxSeq, int32_t numKVHeads, int32_t headDim)
{
    rt::HybridCacheManager::Config cfg{};
    cfg.layerTypes.assign(numLayers, rt::HybridCacheManager::LayerType::kAttention);
    std::vector<rt::KVLayerConfig> layers(numLayers, rt::KVLayerConfig{numKVHeads, headDim});
    cfg.kvConfig = rt::KVCacheManager::Config{numLayers, maxBatch, maxSeq, layers, DataType::kHALF};
    cfg.mambaConfig = rt::MambaCacheManager::Config{};
    cfg.mambaConfig.numRecurrentLayers = 0;
    cfg.mambaConfig.maxBatchSize = maxBatch;
    cfg.maxBatchSize = maxBatch;
    return cfg;
}

} // namespace

TEST(SysPromptCachePagedTest, CaptureRestoreRoundTripNhdAddressing)
{
    cudaStream_t stream{nullptr};

    int32_t const maxBatch = 3;
    int32_t const numLayers = 2;
    int32_t const maxSeq = 64; // capPadded -> 128
    int32_t const numKVHeads = 2;
    int32_t const headDim = 64;
    int32_t const capturedSeq = 40;
    int32_t const captureSlot = 2; // non-zero row to exercise the batchIdx offset
    float const sentinel = -1.0f;

    rt::HybridCacheManager mgr(makeUniformAttnConfig(numLayers, maxBatch, maxSeq, numKVHeads, headDim), stream);

    int32_t const headsTimesDim = numKVHeads * headDim;

    // Seed the NHD pool with a per-token/per-elem pattern in both halves.
    for (int32_t L = 0; L < numLayers; ++L)
    {
        auto [k, v] = mgr.getSeparateKVCache(L);
        seedHalf(k, captureSlot, /*isV=*/0, L, capturedSeq, sentinel);
        seedHalf(v, captureSlot, /*isV=*/1, L, capturedSeq, sentinel);
    }

    // Capture into saved tensors [2, capturedSeq, H, D].
    auto saved = mgr.captureKVCache(captureSlot, capturedSeq, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(saved.size(), static_cast<size_t>(numLayers));
    for (auto const& s : saved)
    {
        auto const& shape = s.getShape();
        ASSERT_EQ(shape[0], 2);
        ASSERT_EQ(shape[1], capturedSeq);
        ASSERT_EQ(shape[2], numKVHeads);
        ASSERT_EQ(shape[3], headDim);
    }

    // Verify the saved tensor holds the NHD pattern: K plane then V plane, each [seqLen, H, D].
    for (int32_t L = 0; L < numLayers; ++L)
    {
        std::vector<half> host(saved[L].getShape().volume());
        CUDA_CHECK(cudaMemcpy(host.data(), saved[L].rawPointer(),
            static_cast<size_t>(saved[L].getShape().volume()) * sizeof(half), cudaMemcpyDeviceToHost));
        int64_t const planeElems = static_cast<int64_t>(capturedSeq) * headsTimesDim;
        for (int32_t isV = 0; isV < 2; ++isV)
        {
            for (int32_t t = 0; t < capturedSeq; ++t)
            {
                for (int32_t e = 0; e < headsTimesDim; ++e)
                {
                    auto got
                        = host[static_cast<size_t>(isV * planeElems + static_cast<int64_t>(t) * headsTimesDim + e)];
                    ASSERT_TRUE(isclose(got, __float2half(patternVal(isV, L, t, e)), 1e-2f, 1e-2f))
                        << "saved L=" << L << " isV=" << isV << " token=" << t << " elem=" << e
                        << " got=" << __half2float(got);
                }
            }
        }
    }

    // Clobber the captured row in the pool, then restore.
    for (int32_t L = 0; L < numLayers; ++L)
    {
        auto [k, v] = mgr.getSeparateKVCache(L);
        fillRow(k, captureSlot, 7777.0f);
        fillRow(v, captureSlot, 7777.0f);
    }

    mgr.restoreKVCache(saved, captureSlot, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // The first capturedSeq tokens must match the seeded pattern again in both halves; tokens beyond
    // the prefix must remain the 7777 clobber (restore only overwrites [0, capturedSeq)).
    int32_t const capPadded = 128;
    for (int32_t L = 0; L < numLayers; ++L)
    {
        auto [k, v] = mgr.getSeparateKVCache(L);
        auto kHost = readRow(k, captureSlot);
        auto vHost = readRow(v, captureSlot);
        for (int32_t t = 0; t < capturedSeq; ++t)
        {
            for (int32_t e = 0; e < headsTimesDim; ++e)
            {
                size_t const idx = static_cast<size_t>(static_cast<int64_t>(t) * headsTimesDim + e);
                ASSERT_TRUE(isclose(kHost[idx], __float2half(patternVal(0, L, t, e)), 1e-2f, 1e-2f))
                    << "restored K L=" << L << " token=" << t << " elem=" << e << " got=" << __half2float(kHost[idx]);
                ASSERT_TRUE(isclose(vHost[idx], __float2half(patternVal(1, L, t, e)), 1e-2f, 1e-2f))
                    << "restored V L=" << L << " token=" << t << " elem=" << e << " got=" << __half2float(vHost[idx]);
            }
        }
        // A token past the prefix is left untouched (still the clobber value).
        size_t const pastIdx = static_cast<size_t>(static_cast<int64_t>(capturedSeq) * headsTimesDim);
        ASSERT_LT(static_cast<int64_t>(capturedSeq), capPadded);
        ASSERT_TRUE(isclose(kHost[pastIdx], __float2half(7777.0f), 1e-2f, 1e-2f))
            << "K prefix overran into untouched region at L=" << L;
        ASSERT_TRUE(isclose(vHost[pastIdx], __float2half(7777.0f), 1e-2f, 1e-2f))
            << "V prefix overran into untouched region at L=" << L;
    }
}
