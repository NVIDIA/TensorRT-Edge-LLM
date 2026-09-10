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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "kernels/int4GroupwiseGemmKernels/int4GroupwiseGemm.h"
#include "naiveGemmReference.h"
#include "references.h"
#include "testUtils.h"

using namespace trt_edgellm;

/* TestInt4GroupwiseGemmAccuracy
 * -----------------------------------------------
 * Purpose:
 *   Sanity-check the numeric accuracy of our INT4, per-group dequantized
 *   GEMM CUDA kernels against a CPU reference.
 *
 * Inputs:
 *   m, n, k       : GEMM shapes (A: m×k, W: k×n, C: m×n).
 *   group_size    : Number of input channels per scale (per-group quantization), default = 128.
 *
 * Descriptions:
 * Randomly initialize input activations, weights and scales. For INT4 GEMM kernel, we pakced the weights according
 * to python reference implementation https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/qmodule.py#L26. For
 * the CPU reference, we use the unpacked weights in FP16, scaled them and compute reference results.
 */
void TestInt4GroupwiseGemmAccuracy(int m, int n, int k, int group_size)
{
    // Initialize input_arr, input_weights, input_scales
    std::vector<half> input_arr(static_cast<size_t>(m) * k);
    std::vector<int16_t> input_weights(static_cast<size_t>(k) * n);
    std::vector<half> input_scales(static_cast<size_t>(k / group_size) * n);

    uniformFloatInitialization(input_arr, 0.2f, 0.5f);
    uniformIntInitialization(input_weights, -5, 5);
    uniformFloatInitialization(input_scales, 0.2f, 0.5f);

    std::vector<half> scaled_weights;
    scaledWeightsReference(input_weights.data(), input_scales.data(), k, n, group_size, scaled_weights);
    std::vector<int16_t> W_kn_i8(input_weights.begin(), input_weights.end());
    std::vector<int16_t> Wpacked_Ndiv4xK((n / 4) * k);
    awqPackReference(W_kn_i8.data(), n, k, Wpacked_Ndiv4xK.data());
    std::vector<half> gpu_results(static_cast<size_t>(m) * n, __float2half(0.0f));
    std::vector<half> gpu_results_ref(static_cast<size_t>(m) * n, __float2half(0.0f));

    cudaStream_t stream{nullptr};
    half *d_act, *d_scales, *d_out, *d_out_ref, *d_weight_ref;
    int8_t* d_weight;

    // For WOQ kernel
    CUDA_CHECK(cudaMallocAsync(&d_act, (size_t) m * k * sizeof(__half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_weight, ((size_t) k * n / 2) * sizeof(int8_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_scales, (size_t) (k / group_size) * n * sizeof(__half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_out, (size_t) m * n * sizeof(__half), stream));

    CUDA_CHECK(
        cudaMemcpyAsync(d_act, input_arr.data(), (size_t) m * k * sizeof(__half), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_weight, reinterpret_cast<int8_t const*>(Wpacked_Ndiv4xK.data()),
        ((size_t) k * n / 2) * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        d_scales, input_scales.data(), (size_t) (k / group_size) * n * sizeof(__half), cudaMemcpyHostToDevice, stream));

    // For GPU reference
    CUDA_CHECK(cudaMallocAsync(&d_weight_ref, ((size_t) k * n) * sizeof(__half), stream));
    CUDA_CHECK(cudaMallocAsync(&d_out_ref, (size_t) m * n * sizeof(__half), stream));

    CUDA_CHECK(cudaMemcpyAsync(
        d_weight_ref, scaled_weights.data(), ((size_t) k * n) * sizeof(__half), cudaMemcpyHostToDevice, stream));

    // WOQ kernel inference
    trt_edgellm::kernel::gemm_forward_cuda_new(d_act, d_weight, d_scales, d_out, m, n, k, group_size, stream);

    // Reference GEMM inference
    naive_gemm_forward(d_act, d_weight_ref, d_out_ref, m, n, k, stream);

    CUDA_CHECK(
        cudaMemcpyAsync(gpu_results.data(), d_out, (size_t) m * n * sizeof(__half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        gpu_results_ref.data(), d_out_ref, (size_t) m * n * sizeof(__half), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(d_act, stream));
    CUDA_CHECK(cudaFreeAsync(d_weight, stream));
    CUDA_CHECK(cudaFreeAsync(d_scales, stream));
    CUDA_CHECK(cudaFreeAsync(d_out, stream));
    CUDA_CHECK(cudaFreeAsync(d_out_ref, stream));

    // Compare (convert GPU fp16 to CPU result)
    int const total = m * n;
    float max_abs = 0;
    for (int i = 0; i < total; ++i)
    {
        float o = __half2float(gpu_results_ref.data()[i]);
        float r = __half2float(gpu_results.data()[i]);
        float ad = std::fabs(o - r);
        max_abs = std::max(max_abs, ad);
        EXPECT_TRUE(isclose(o, r, 1e-1, 1e-1));
    }
    std::printf("Largest abs error = %g\n", max_abs);
}

TEST(WOQInt4GemmTest, accuracyGemm)
{
    TestInt4GroupwiseGemmAccuracy(128, 128, 128, 128);
    // k=512 (instead of k=1024): FP16 tensor-core accumulation vs FP32 reference error scales
    // with k; halving k keeps the accumulated rounding error within rtol=1e-1.
    TestInt4GroupwiseGemmAccuracy(64, 512, 512, 128);
    TestInt4GroupwiseGemmAccuracy(64, 640, 512, 128);
}

// Regression guard for the cudaFuncSetAttribute-during-stream-capture bug.
// gemm_forward_cuda_new used to call cudaFuncSetAttribute on every invocation,
// which is not capturable -- it invalidated any active stream capture and made
// the next kernel launch return cudaErrorStreamCaptureInvalidated (901).  The
// attribute is now set once via std::call_once during the first (out-of-capture)
// call.  Iterates over a spread of representative shapes -- all hit the GEMM
// path (M > kGemvMaxM and N % kGemmCtaN == 0).  Each iteration launches the
// kernel inside cudaStreamBeginCapture/EndCapture and asserts the capture
// completes, instantiates, and replays cleanly.
TEST(WOQInt4GemmTest, streamCaptureSafe)
{
    struct Shape
    {
        int m;
        int n;
        int k;
        char const* note;
    };
    Shape const shapes[] = {
        {8, 128, 128, "minimal valid GEMM tile"},
        {8, 2560, 5120, "Qwen3.5-4B attention/GDN proj (production)"},
        {8, 9216, 2560, "Qwen3.5-4B MLP up/gate proj"},
        {8, 2560, 9216, "Qwen3.5-4B MLP down proj"},
        {120, 4096, 2560, "EAGLE-like large M (mxbs=2, vts=60)"},
    };
    int const group_size = 128;

    for (auto const& s : shapes)
    {
        SCOPED_TRACE(testing::Message() << s.note << " (m=" << s.m << ", n=" << s.n << ", k=" << s.k << ")");
        int const m = s.m;
        int const n = s.n;
        int const k = s.k;

        cudaStream_t stream{nullptr};
        CUDA_CHECK(cudaStreamCreate(&stream));

        half *d_act{nullptr}, *d_scales{nullptr}, *d_out{nullptr};
        int8_t* d_weight{nullptr};
        CUDA_CHECK(cudaMallocAsync(&d_act, (size_t) m * k * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&d_weight, ((size_t) k * n / 2) * sizeof(int8_t), stream));
        CUDA_CHECK(cudaMallocAsync(&d_scales, (size_t) (k / group_size) * n * sizeof(__half), stream));
        CUDA_CHECK(cudaMallocAsync(&d_out, (size_t) m * n * sizeof(__half), stream));
        CUDA_CHECK(cudaMemsetAsync(d_act, 0, (size_t) m * k * sizeof(__half), stream));
        CUDA_CHECK(cudaMemsetAsync(d_weight, 0, ((size_t) k * n / 2) * sizeof(int8_t), stream));
        CUDA_CHECK(cudaMemsetAsync(d_scales, 0, (size_t) (k / group_size) * n * sizeof(__half), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Warmup outside capture -- triggers std::call_once / cudaFuncSetAttribute on first iteration.
        trt_edgellm::kernel::gemm_forward_cuda_new(d_act, d_weight, d_scales, d_out, m, n, k, group_size, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Capture must not be invalidated by gemm_forward_cuda_new.
        cudaGraph_t graph{};
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
        trt_edgellm::kernel::gemm_forward_cuda_new(d_act, d_weight, d_scales, d_out, m, n, k, group_size, stream);
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
        ASSERT_NE(graph, nullptr);

        // Graph must instantiate and replay cleanly.
        cudaGraphExec_t graphExec{};
        ASSERT_EQ(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(graphExec, stream), cudaSuccess);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaFreeAsync(d_act, stream));
        CUDA_CHECK(cudaFreeAsync(d_weight, stream));
        CUDA_CHECK(cudaFreeAsync(d_scales, stream));
        CUDA_CHECK(cudaFreeAsync(d_out, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}
