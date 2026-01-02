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

#include <cuda_fp16.h>
#include <stdexcept>

__host__ __device__ __forceinline__ int idx2(int r, int c, int ld)
{
    return r * ld + c;
}

__global__ void naive_gemm_kernel(half const* A, half const* B, half* C, int M, int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
    {
        float a = __half2float(A[idx2(m, k, K)]);
        float b = __half2float(B[idx2(k, n, N)]);
        acc += a * b;
    }
    C[idx2(m, n, N)] = __float2half(acc);
}

void naive_gemm_forward(half* in_feats, half* weights, half* out_feats, int m, int n, int k, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    naive_gemm_kernel<<<grid, block, 0, stream>>>(in_feats, weights, out_feats, m, n, k);
}
