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

#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "common/cudaUtils.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include "multimodal/cosmos3/cosmos3EdgeViTRunner.h"
#include "testUtils.h"

using namespace trt_edgellm;

namespace
{

// Hugging Face Cosmos3EdgeImageProcessor.patchify for one frame: 2x2 merge-block-major patches, each flattened
// channel-last as (ph, pw, c). This is the column order of the checkpoint's patch-embedding weight.
std::vector<float> hfPatchify(std::vector<half> const& image, int32_t H, int32_t W, int32_t C, int32_t P, int32_t M)
{
    int32_t const gh = H / P;
    int32_t const gw = W / P;
    std::vector<float> out(static_cast<size_t>(gh) * gw * P * P * C);
    size_t n = 0;
    for (int32_t bh = 0; bh < gh / M; ++bh)
        for (int32_t bw = 0; bw < gw / M; ++bw)
            for (int32_t mh = 0; mh < M; ++mh)
                for (int32_t mw = 0; mw < M; ++mw)
                    for (int32_t ph = 0; ph < P; ++ph)
                        for (int32_t pw = 0; pw < P; ++pw)
                            for (int32_t c = 0; c < C; ++c)
                            {
                                int32_t const y = (bh * M + mh) * P + ph;
                                int32_t const x = (bw * M + mw) * P + pw;
                                out[n++] = __half2float(image[(static_cast<size_t>(y) * W + x) * C + c]);
                            }
    return out;
}

} // namespace

// The runtime patch layout and the exporter's weight adaptation must compose to the checkpoint's embedding:
//   runtime_patches · W_exported^T == hf_patches · W_hf^T
// where W_exported[d, c*P*P + ph*P + pw] = W_hf[d, (ph*P + pw)*C + c] (_adapt_patch_embedding_weight_for_chw_input).
TEST(Cosmos3PatchLayout, RuntimeLayoutMatchesExporterWeightAdaptation)
{
    constexpr int32_t H = 64, W = 96, C = 3, P = 16, M = 2, D = 8;
    constexpr int32_t K = C * P * P;
    cudaStream_t stream{nullptr};

    std::vector<half> image(static_cast<size_t>(H) * W * C);
    uniformFloatInitialization<half>(image, -1.f, 1.f);
    std::mt19937 rng(232);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> wHf(static_cast<size_t>(D) * K);
    for (auto& v : wHf)
    {
        v = dist(rng);
    }
    std::vector<float> wExported(wHf.size());
    for (int32_t d = 0; d < D; ++d)
        for (int32_t ph = 0; ph < P; ++ph)
            for (int32_t pw = 0; pw < P; ++pw)
                for (int32_t c = 0; c < C; ++c)
                {
                    wExported[static_cast<size_t>(d) * K + c * P * P + ph * P + pw]
                        = wHf[static_cast<size_t>(d) * K + (ph * P + pw) * C + c];
                }

    rt::Tensor imageDevice({1, H, W, C}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    CUDA_CHECK(cudaMemcpy(imageDevice.rawPointer(), image.data(), image.size() * sizeof(half), cudaMemcpyHostToDevice));
    int32_t const numPatches = (H / P) * (W / P);
    rt::Tensor patchesDevice({numPatches, K}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    kernel::transposeToPatchQwenViT(imageDevice, patchesDevice, 0, /*temporalPatchSize=*/1, P, M,
        /*temporalFirst=*/false, rt::Cosmos3EdgeViTRunner::kPatchChannelLast, stream);
    std::vector<half> patches(static_cast<size_t>(numPatches) * K);
    CUDA_CHECK(
        cudaMemcpy(patches.data(), patchesDevice.rawPointer(), patches.size() * sizeof(half), cudaMemcpyDeviceToHost));

    std::vector<float> const ref = hfPatchify(image, H, W, C, P, M);
    for (int32_t n = 0; n < numPatches; ++n)
    {
        for (int32_t d = 0; d < D; ++d)
        {
            double got = 0.0;
            double want = 0.0;
            for (int32_t k = 0; k < K; ++k)
            {
                got += static_cast<double>(__half2float(patches[static_cast<size_t>(n) * K + k]))
                    * wExported[static_cast<size_t>(d) * K + k];
                want += static_cast<double>(ref[static_cast<size_t>(n) * K + k]) * wHf[static_cast<size_t>(d) * K + k];
            }
            ASSERT_NEAR(got, want, 1e-3) << "patch " << n << " out " << d;
        }
    }
}
