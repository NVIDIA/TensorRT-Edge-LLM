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

#include "common/checkMacros.h"
#include "common/stringUtils.h"
#include "moeActivationKernels.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

using format::fmtstr;

// ====================== Helper Functions ======================

// 128-bit aligned vector type for coalesced loads and stores of eight FP16 or BF16 elements.
constexpr int32_t kElemPerVec = 8;
constexpr int32_t kGatedInputMultiplier = 2;

template <typename T>
struct alignas(16) Vec8
{
    T data[kElemPerVec];
    __device__ __host__ T& operator[](int32_t idx)
    {
        return data[idx];
    }
    __device__ __host__ T const& operator[](int32_t idx) const
    {
        return data[idx];
    }
};

template <typename T>
struct ActivationMath;

template <typename T>
__device__ __forceinline__ float toFloat(T value);

template <>
__device__ __forceinline__ float toFloat(half value)
{
    return __half2float(value);
}

template <>
__device__ __forceinline__ float toFloat(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <>
struct ActivationMath<half>
{
    static __device__ __forceinline__ half swiGlu(half gate, half up)
    {
        float const gateFloat = __half2float(gate);
        half const activatedGate = __float2half(gateFloat / (1.0F + expf(-gateFloat)));
        return __hmul(activatedGate, up);
    }

    static __device__ __forceinline__ half swiGlu(half gate, half up, float upScale)
    {
        float const gateFloat = __half2float(gate);
        half const activatedGate = __float2half(gateFloat / (1.0F + expf(-gateFloat)));
        return __float2half(__half2float(activatedGate) * __half2float(up) * upScale);
    }

    static __device__ __forceinline__ half relu2(half input)
    {
        float const inputFloat = __half2float(input);
        float const relu = inputFloat < 0.0F ? 0.0F : inputFloat;
        return __float2half_rn(relu * relu);
    }

    static __device__ __forceinline__ half geGlu(half gate, half up)
    {
        float const gateFloat = __half2float(gate);
        float const cdf
            = 0.5F * (1.0F + tanhf(0.7978845608F * (gateFloat + 0.044715F * gateFloat * gateFloat * gateFloat)));
        return __float2half(gateFloat * cdf * __half2float(up));
    }

    static __device__ __forceinline__ half geGlu(half gate, half up, float upScale)
    {
        float const gateFloat = __half2float(gate);
        float const cdf
            = 0.5F * (1.0F + tanhf(0.7978845608F * (gateFloat + 0.044715F * gateFloat * gateFloat * gateFloat)));
        return __float2half(gateFloat * cdf * __half2float(up) * upScale);
    }
};

template <>
struct ActivationMath<__nv_bfloat16>
{
    static __device__ __forceinline__ __nv_bfloat16 swiGlu(__nv_bfloat16 gate, __nv_bfloat16 up)
    {
        float const gateFloat = __bfloat162float(gate);
        float const upFloat = __bfloat162float(up);
        float const activatedGate = gateFloat / (1.0F + expf(-gateFloat));
        return __float2bfloat16_rn(activatedGate * upFloat);
    }

    static __device__ __forceinline__ __nv_bfloat16 swiGlu(__nv_bfloat16 gate, __nv_bfloat16 up, float upScale)
    {
        float const gateFloat = __bfloat162float(gate);
        float const upFloat = __bfloat162float(up);
        float const activatedGate = gateFloat / (1.0F + expf(-gateFloat));
        return __float2bfloat16_rn(activatedGate * upFloat * upScale);
    }

    static __device__ __forceinline__ __nv_bfloat16 relu2(__nv_bfloat16 input)
    {
        float const inputFloat = __bfloat162float(input);
        float const relu = inputFloat < 0.0F ? 0.0F : inputFloat;
        return __float2bfloat16_rn(relu * relu);
    }

    static __device__ __forceinline__ __nv_bfloat16 geGlu(__nv_bfloat16 gate, __nv_bfloat16 up)
    {
        float const gateFloat = __bfloat162float(gate);
        float const upFloat = __bfloat162float(up);
        float const cdf
            = 0.5F * (1.0F + tanhf(0.7978845608F * (gateFloat + 0.044715F * gateFloat * gateFloat * gateFloat)));
        return __float2bfloat16_rn(gateFloat * cdf * upFloat);
    }

    static __device__ __forceinline__ __nv_bfloat16 geGlu(__nv_bfloat16 gate, __nv_bfloat16 up, float upScale)
    {
        float const gateFloat = __bfloat162float(gate);
        float const upFloat = __bfloat162float(up);
        float const cdf
            = 0.5F * (1.0F + tanhf(0.7978845608F * (gateFloat + 0.044715F * gateFloat * gateFloat * gateFloat)));
        return __float2bfloat16_rn(gateFloat * cdf * upFloat * upScale);
    }
};

// ====================== SwiGLU Kernel ======================

// Input layout: [numTokens, 2 * intermediateDim] with gate in the first half and up in the second.
// The kernel receives a single base pointer typed as Vec8 and computes per-token gate/up offsets.
// numVecsPerRow = intermediateDim / kElemPerVec (the number of vecs covering one gate or up segment).
template <typename T, bool kScaleUp = false>
__global__ void swiGluKernel(Vec8<T> const* __restrict__ input, Vec8<T>* __restrict__ output, int64_t numVecsPerRow,
    int64_t numTokens, T const* projectionGlobalScales = nullptr, int32_t const* expertIndices = nullptr,
    int32_t numExperts = 0)
{
    int64_t const tokenIdx = blockIdx.x;
    if (tokenIdx >= numTokens)
    {
        return;
    }

    // Per-token vec offsets: gate and up are stored contiguously with stride 2 * numVecsPerRow.
    int64_t const gateStart = tokenIdx * kGatedInputMultiplier * numVecsPerRow;
    int64_t const upStart = gateStart + numVecsPerRow;
    int64_t const outStart = tokenIdx * numVecsPerRow;
    float upScale = 1.0F;
    if constexpr (kScaleUp)
    {
        int32_t const expert = expertIndices[tokenIdx];
        upScale = toFloat(projectionGlobalScales[numExperts + expert]) / toFloat(projectionGlobalScales[expert]);
    }

    int64_t const tid = threadIdx.x;
    int64_t const stride = blockDim.x;

    for (int64_t i = tid; i < numVecsPerRow; i += stride)
    {
        Vec8<T> const gateVec = input[gateStart + i];
        Vec8<T> const upVec = input[upStart + i];
        Vec8<T> outVec;

#pragma unroll
        for (int32_t j = 0; j < kElemPerVec; j++)
        {
            if constexpr (kScaleUp)
            {
                outVec[j] = ActivationMath<T>::swiGlu(gateVec[j], upVec[j], upScale);
            }
            else
            {
                outVec[j] = ActivationMath<T>::swiGlu(gateVec[j], upVec[j]);
            }
        }

        output[outStart + i] = outVec;
    }
}

template <typename T>
__global__ void relu2Kernel(
    Vec8<T> const* __restrict__ input, Vec8<T>* __restrict__ output, int64_t numVecsPerRow, int64_t numTokens)
{
    int64_t const tokenIdx = blockIdx.x;
    if (tokenIdx >= numTokens)
    {
        return;
    }

    int64_t const rowStart = tokenIdx * numVecsPerRow;
    int64_t const tid = threadIdx.x;
    int64_t const stride = blockDim.x;
    for (int64_t i = tid; i < numVecsPerRow; i += stride)
    {
        Vec8<T> const inputVec = input[rowStart + i];
        Vec8<T> outVec;

#pragma unroll
        for (int32_t j = 0; j < kElemPerVec; j++)
        {
            outVec[j] = ActivationMath<T>::relu2(inputVec[j]);
        }

        output[rowStart + i] = outVec;
    }
}

// ====================== GeGLU Kernel ======================

template <typename T, bool kScaleUp = false>
__global__ void geGluKernel(Vec8<T> const* __restrict__ input, Vec8<T>* __restrict__ output, int64_t numVecsPerRow,
    int64_t numTokens, T const* projectionGlobalScales = nullptr, int32_t const* expertIndices = nullptr,
    int32_t numExperts = 0)
{
    int64_t const tokenIdx = blockIdx.x;
    if (tokenIdx >= numTokens)
    {
        return;
    }

    int64_t const gateStart = tokenIdx * kGatedInputMultiplier * numVecsPerRow;
    int64_t const upStart = gateStart + numVecsPerRow;
    int64_t const outStart = tokenIdx * numVecsPerRow;
    float upScale = 1.0F;
    if constexpr (kScaleUp)
    {
        int32_t const expert = expertIndices[tokenIdx];
        upScale = toFloat(projectionGlobalScales[numExperts + expert]) / toFloat(projectionGlobalScales[expert]);
    }

    int64_t const tid = threadIdx.x;
    int64_t const stride = blockDim.x;

    for (int64_t i = tid; i < numVecsPerRow; i += stride)
    {
        Vec8<T> gateVec = input[gateStart + i];
        Vec8<T> upVec = input[upStart + i];
        Vec8<T> outVec;

#pragma unroll
        for (int32_t j = 0; j < kElemPerVec; j++)
        {
            if constexpr (kScaleUp)
            {
                outVec[j] = ActivationMath<T>::geGlu(gateVec[j], upVec[j], upScale);
            }
            else
            {
                outVec[j] = ActivationMath<T>::geGlu(gateVec[j], upVec[j]);
            }
        }

        output[outStart + i] = outVec;
    }
}

// ====================== Launch Helper ======================

template <typename T>
void launchMoeActivation(rt::Tensor const& input, rt::Tensor& output, int64_t numTokens, int64_t intermediateDim,
    int32_t activationType, char const* inputName, cudaStream_t stream)
{
    using VectorType = Vec8<T>;
    auto const* inputRawPtr = input.dataPointer<T>();
    auto* outputRawPtr = output.dataPointer<T>();
    check::check(reinterpret_cast<uintptr_t>(inputRawPtr) % alignof(VectorType) == 0,
        fmtstr("%s pointer must be 16-byte aligned for vectorized access", inputName));
    check::check(reinterpret_cast<uintptr_t>(outputRawPtr) % alignof(VectorType) == 0,
        "output pointer must be 16-byte aligned for vectorized access");

    int64_t const numVecsPerRow = intermediateDim / kElemPerVec;
    auto const* inputVec = reinterpret_cast<VectorType const*>(inputRawPtr);
    auto* outputVec = reinterpret_cast<VectorType*>(outputRawPtr);
    int64_t const blocks = numTokens;
    int32_t const threads = 1024;
    if (activationType == MoeActivationType::kMoeSwiGlu)
    {
        swiGluKernel<T><<<blocks, threads, 0, stream>>>(inputVec, outputVec, numVecsPerRow, numTokens);
    }
    else if (activationType == MoeActivationType::kMoeRelu2)
    {
        relu2Kernel<T><<<blocks, threads, 0, stream>>>(inputVec, outputVec, numVecsPerRow, numTokens);
    }
    else if (activationType == MoeActivationType::kMoeGeGlu)
    {
        geGluKernel<T><<<blocks, threads, 0, stream>>>(inputVec, outputVec, numVecsPerRow, numTokens);
    }
}

// ====================== Public API ======================

void moeActivation(rt::Tensor const& input, rt::Tensor& output, int64_t numTokens, int64_t intermediateDim,
    int32_t activationType, cudaStream_t stream)
{
    check::check(activationType == MoeActivationType::kMoeSwiGlu || activationType == MoeActivationType::kMoeRelu2
            || activationType == MoeActivationType::kMoeGeGlu,
        fmtstr("Unsupported MoE activation type %d; expected %d (SwiGLU), %d (ReLU2), or %d (GeGLU)", activationType,
            MoeActivationType::kMoeSwiGlu, MoeActivationType::kMoeRelu2, MoeActivationType::kMoeGeGlu));

    bool const isGated
        = (activationType == MoeActivationType::kMoeSwiGlu || activationType == MoeActivationType::kMoeGeGlu);
    auto const inputShape = input.getShape();
    auto const outputShape = output.getShape();
    int64_t const expectedInputDim = isGated ? kGatedInputMultiplier * intermediateDim : intermediateDim;

    check::check(inputShape.getNumDims() == 2, "input must be a 2D tensor");
    check::check(outputShape.getNumDims() == 2, "output must be a 2D tensor");
    check::check(inputShape[0] == numTokens, "input first dimension must match numTokens");
    check::check(inputShape[1] == expectedInputDim,
        fmtstr("input second dimension must be %ld for activation type %d", expectedInputDim, activationType));
    check::check(outputShape[0] == numTokens, "output first dimension must match numTokens");
    check::check(outputShape[1] == intermediateDim,
        fmtstr("output second dimension must be intermediateDim = %ld", intermediateDim));

    nvinfer1::DataType const dataType = input.getDataType();
    check::check(
        dataType == nvinfer1::DataType::kHALF || dataType == nvinfer1::DataType::kBF16, "input must be FP16 or BF16");
    check::check(output.getDataType() == dataType, "input and output data types must match");
    check::check(input.getDeviceType() == rt::DeviceType::kGPU, "input must be on GPU");
    check::check(output.getDeviceType() == rt::DeviceType::kGPU, "output must be on GPU");

    check::check(intermediateDim % kElemPerVec == 0,
        fmtstr("intermediateDim (%ld) must be a multiple of %d for vectorized access", intermediateDim, kElemPerVec));

    if (dataType == nvinfer1::DataType::kHALF)
    {
        launchMoeActivation<half>(input, output, numTokens, intermediateDim, activationType, "input", stream);
    }
    else
    {
        launchMoeActivation<__nv_bfloat16>(input, output, numTokens, intermediateDim, activationType, "input", stream);
    }
}

template <typename T>
void launchScaledGatedActivation(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& projectionGlobalScales,
    rt::Tensor const& expertIndices, int64_t numTokens, int64_t intermediateDim, int32_t activationType,
    cudaStream_t stream)
{
    int32_t const numExperts = static_cast<int32_t>(projectionGlobalScales.getShape()[1]);
    int64_t const numVecsPerRow = intermediateDim / kElemPerVec;
    auto const* inputVec = reinterpret_cast<Vec8<T> const*>(input.dataPointer<T>());
    auto* outputVec = reinterpret_cast<Vec8<T>*>(output.dataPointer<T>());
    T const* scales = projectionGlobalScales.dataPointer<T>();
    int32_t const* experts = expertIndices.dataPointer<int32_t>();
    int32_t constexpr threads = 1024;
    if (activationType == MoeActivationType::kMoeSwiGlu)
    {
        swiGluKernel<T, true><<<numTokens, threads, 0, stream>>>(
            inputVec, outputVec, numVecsPerRow, numTokens, scales, experts, numExperts);
    }
    else
    {
        geGluKernel<T, true><<<numTokens, threads, 0, stream>>>(
            inputVec, outputVec, numVecsPerRow, numTokens, scales, experts, numExperts);
    }
}

void moeScaledGatedActivation(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& projectionGlobalScales,
    rt::Tensor const& expertIndices, int64_t numTokens, int64_t intermediateDim, int32_t activationType,
    cudaStream_t stream)
{
    check::check(activationType == MoeActivationType::kMoeSwiGlu || activationType == MoeActivationType::kMoeGeGlu,
        "scaled gated activation requires SwiGLU or GeGLU");
    check::check(
        input.getShape() == rt::Coords{numTokens, 2 * intermediateDim}, "scaled gated activation input shape mismatch");
    check::check(
        output.getShape() == rt::Coords{numTokens, intermediateDim}, "scaled gated activation output shape mismatch");
    check::check(projectionGlobalScales.getShape().getNumDims() == 2 && projectionGlobalScales.getShape()[0] == 2
            && projectionGlobalScales.getShape()[1] > 0,
        "scaled gated activation requires [2, num_experts] global scales");
    check::check(
        expertIndices.getShape().volume() == numTokens && expertIndices.getDataType() == nvinfer1::DataType::kINT32,
        "scaled gated activation expert indices mismatch");
    check::check(
        input.getDataType() == output.getDataType() && input.getDataType() == projectionGlobalScales.getDataType(),
        "scaled gated activation data types must match");
    check::check(input.getDeviceType() == rt::DeviceType::kGPU && output.getDeviceType() == rt::DeviceType::kGPU
            && projectionGlobalScales.getDeviceType() == rt::DeviceType::kGPU
            && expertIndices.getDeviceType() == rt::DeviceType::kGPU,
        "scaled gated activation tensors must be on GPU");
    check::check(intermediateDim % kElemPerVec == 0,
        fmtstr("intermediateDim (%ld) must be a multiple of %d", intermediateDim, kElemPerVec));

    if (input.getDataType() == nvinfer1::DataType::kHALF)
    {
        launchScaledGatedActivation<half>(
            input, output, projectionGlobalScales, expertIndices, numTokens, intermediateDim, activationType, stream);
    }
    else
    {
        check::check(input.getDataType() == nvinfer1::DataType::kBF16, "scaled gated activation supports FP16 or BF16");
        launchScaledGatedActivation<__nv_bfloat16>(
            input, output, projectionGlobalScales, expertIndices, numTokens, intermediateDim, activationType, stream);
    }
}

} // namespace kernel
} // namespace trt_edgellm
