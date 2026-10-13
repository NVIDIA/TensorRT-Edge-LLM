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

#include "kernels/speculative/dflash2GroupedDynamicConv.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

template <typename T>
struct PackedTraits;

template <>
struct PackedTraits<half>
{
    using Packed = half2;

    __device__ static float2 toFloat2(Packed value)
    {
        return __half22float2(value);
    }

    __device__ static Packed fromFloat2(float2 value)
    {
        constexpr float kHalfMax = 65504.0F;
        value.x = fminf(kHalfMax, fmaxf(-kHalfMax, value.x));
        value.y = fminf(kHalfMax, fmaxf(-kHalfMax, value.y));
        return __floats2half2_rn(value.x, value.y);
    }

    __device__ static float toFloat(half value)
    {
        return __half2float(value);
    }
};

template <>
struct PackedTraits<__nv_bfloat16>
{
    using Packed = __nv_bfloat162;

    __device__ static float2 toFloat2(Packed value)
    {
        return __bfloat1622float2(value);
    }

    __device__ static Packed fromFloat2(float2 value)
    {
        return __floats2bfloat162_rn(value.x, value.y);
    }

    __device__ static float toFloat(__nv_bfloat16 value)
    {
        return __bfloat162float(value);
    }
};

template <typename T, int32_t TAPS, bool FUSE_RESIDUAL>
__global__ void groupedDynamicConvKernel(T const* __restrict__ hidden, T const* __restrict__ delta,
    T const* __restrict__ base, float const* __restrict__ residual, void* __restrict__ output, int32_t tokens,
    int32_t hiddenSize, int32_t blockSize, int32_t groupSize)
{
    using Traits = PackedTraits<T>;
    using Packed = typename Traits::Packed;
    int32_t const token = static_cast<int32_t>(blockIdx.y);
    int32_t const packedHidden = hiddenSize / 2;
    int32_t const numGroups = hiddenSize / groupSize;
    int32_t const position = token % blockSize;
    auto const* hidden2 = reinterpret_cast<Packed const*>(hidden);
    auto const* base2 = reinterpret_cast<Packed const*>(base);

    for (int32_t packedChannel = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
        packedChannel < packedHidden; packedChannel += static_cast<int32_t>(gridDim.x * blockDim.x))
    {
        int32_t const channel = packedChannel * 2;
        int32_t const group = channel / groupSize;
        float2 sum = make_float2(0.0F, 0.0F);
#pragma unroll
        for (int32_t tap = 0; tap < TAPS; ++tap)
        {
            if (position >= tap)
            {
                float const dynamic = Traits::toFloat(delta[(token * TAPS + tap) * numGroups + group]);
                float2 const fixed = Traits::toFloat2(base2[tap * packedHidden + packedChannel]);
                float2 const value = Traits::toFloat2(hidden2[(token - tap) * packedHidden + packedChannel]);
                sum.x = fmaf(fixed.x + dynamic, value.x, sum.x);
                sum.y = fmaf(fixed.y + dynamic, value.y, sum.y);
            }
        }
        if constexpr (FUSE_RESIDUAL)
        {
            auto const* residual2 = reinterpret_cast<float2 const*>(residual);
            auto* output2 = reinterpret_cast<float2*>(output);
            float2 const residualValue = residual2[token * packedHidden + packedChannel];
            sum.x += residualValue.x;
            sum.y += residualValue.y;
            output2[token * packedHidden + packedChannel] = sum;
        }
        else
        {
            auto* output2 = reinterpret_cast<Packed*>(output);
            output2[token * packedHidden + packedChannel] = Traits::fromFloat2(sum);
        }
    }
}

bool isGpuTensor(rt::Tensor const& tensor) noexcept
{
    return tensor.getDeviceType() == rt::DeviceType::kGPU && tensor.rawPointer() != nullptr;
}

bool hasGroupedDynamicConvContract(rt::Tensor const& hidden, rt::Tensor const& delta, rt::Tensor const& base,
    rt::OptionalInputTensor residual, rt::Tensor const& output) noexcept
{
    rt::Coords const hiddenShape = hidden.getShape();
    rt::Coords const deltaShape = delta.getShape();
    rt::Coords const baseShape = base.getShape();
    int32_t const hiddenRank = hiddenShape.getNumDims();
    if (hiddenRank < 2 || hiddenRank > 4 || deltaShape.getNumDims() != hiddenRank + 1 || baseShape.getNumDims() != 2)
    {
        return false;
    }
    constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
    if (hiddenShape[hiddenRank - 1] <= 0 || hiddenShape[hiddenRank - 1] > kInt32Max || hiddenShape[hiddenRank - 2] <= 0
        || hiddenShape[hiddenRank - 2] > kInt32Max || baseShape[0] <= 0 || baseShape[0] > kInt32Max
        || deltaShape[hiddenRank] <= 0 || deltaShape[hiddenRank] > kInt32Max)
    {
        return false;
    }
    int32_t const hiddenSize = static_cast<int32_t>(hiddenShape[hiddenRank - 1]);
    int32_t const blockSize = static_cast<int32_t>(hiddenShape[hiddenRank - 2]);
    int32_t const kernelSize = static_cast<int32_t>(baseShape[0]);
    int32_t const groups = static_cast<int32_t>(deltaShape[hiddenRank]);
    if (hiddenSize % 2 != 0 || blockSize > kDFlash2MaxBlockSize || kernelSize > 4 || hiddenSize % groups != 0)
    {
        return false;
    }
    int32_t const groupSize = hiddenSize / groups;
    nvinfer1::DataType const activationType = hidden.getDataType();
    if ((activationType != nvinfer1::DataType::kHALF && activationType != nvinfer1::DataType::kBF16)
        || groupSize % 2 != 0 || delta.getDataType() != activationType || base.getDataType() != activationType
        || baseShape != rt::Coords{kernelSize, hiddenSize} || output.getShape() != hiddenShape || !isGpuTensor(hidden)
        || !isGpuTensor(delta) || !isGpuTensor(base) || !isGpuTensor(output))
    {
        return false;
    }
    for (int32_t dim = 0; dim < hiddenRank - 1; ++dim)
    {
        if (hiddenShape[dim] <= 0 || deltaShape[dim] != hiddenShape[dim])
        {
            return false;
        }
    }
    if (deltaShape[hiddenRank - 1] != kernelSize)
    {
        return false;
    }
    if (hiddenShape.volume() / hiddenSize > std::numeric_limits<int32_t>::max())
    {
        return false;
    }
    if (residual.has_value())
    {
        rt::Tensor const& residualTensor = residual->get();
        return residualTensor.getShape() == hiddenShape && residualTensor.getDataType() == nvinfer1::DataType::kFLOAT
            && output.getDataType() == nvinfer1::DataType::kFLOAT && isGpuTensor(residualTensor);
    }
    return output.getDataType() == activationType;
}

template <typename T>
cudaError_t dispatchTaps(T const* hidden, T const* delta, T const* base, void* output, int32_t tokens,
    float const* residual, int32_t hiddenSize, int32_t blockSize, int32_t kernelSize, int32_t groupSize,
    cudaStream_t stream)
{
    constexpr int32_t kThreads = 256;
    int32_t const packedHidden = hiddenSize / 2;
    dim3 const grid(static_cast<uint32_t>((packedHidden + kThreads - 1) / kThreads), static_cast<uint32_t>(tokens));
#define DFLASH2_LAUNCH(TAPS)                                                                                           \
    if (residual != nullptr)                                                                                           \
        groupedDynamicConvKernel<T, TAPS, true><<<grid, kThreads, 0, stream>>>(                                        \
            hidden, delta, base, residual, output, tokens, hiddenSize, blockSize, groupSize);                          \
    else                                                                                                               \
        groupedDynamicConvKernel<T, TAPS, false><<<grid, kThreads, 0, stream>>>(                                       \
            hidden, delta, base, nullptr, output, tokens, hiddenSize, blockSize, groupSize)
    switch (kernelSize)
    {
    case 1: DFLASH2_LAUNCH(1); break;
    case 2: DFLASH2_LAUNCH(2); break;
    case 3: DFLASH2_LAUNCH(3); break;
    case 4: DFLASH2_LAUNCH(4); break;
    default: return cudaErrorInvalidValue;
    }
#undef DFLASH2_LAUNCH
    return cudaGetLastError();
}

} // namespace

cudaError_t launchDFlash2GroupedDynamicConv(rt::Tensor const& hidden, rt::Tensor const& delta, rt::Tensor const& base,
    rt::OptionalInputTensor residual, rt::Tensor& output, cudaStream_t stream) noexcept
{
    if (!hasGroupedDynamicConvContract(hidden, delta, base, residual, output))
    {
        return cudaErrorInvalidValue;
    }
    rt::Coords const hiddenShape = hidden.getShape();
    rt::Coords const baseShape = base.getShape();
    rt::Coords const deltaShape = delta.getShape();
    int32_t const hiddenRank = hiddenShape.getNumDims();
    int32_t const hiddenSize = static_cast<int32_t>(hiddenShape[hiddenRank - 1]);
    int32_t const blockSize = static_cast<int32_t>(hiddenShape[hiddenRank - 2]);
    int32_t const kernelSize = static_cast<int32_t>(baseShape[0]);
    int32_t const groupSize = hiddenSize / static_cast<int32_t>(deltaShape[hiddenRank]);
    int32_t const tokens = static_cast<int32_t>(hiddenShape.volume() / hiddenSize);
    float const* residualData = residual.has_value() ? residual->get().dataPointer<float>() : nullptr;
    if (hidden.getDataType() == nvinfer1::DataType::kHALF)
    {
        return dispatchTaps(hidden.dataPointer<half>(), delta.dataPointer<half>(), base.dataPointer<half>(),
            output.rawPointer(), tokens, residualData, hiddenSize, blockSize, kernelSize, groupSize, stream);
    }
    if (hidden.getDataType() == nvinfer1::DataType::kBF16)
    {
        return dispatchTaps(hidden.dataPointer<__nv_bfloat16>(), delta.dataPointer<__nv_bfloat16>(),
            base.dataPointer<__nv_bfloat16>(), output.rawPointer(), tokens, residualData, hiddenSize, blockSize,
            kernelSize, groupSize, stream);
    }
    return cudaErrorInvalidValue;
}

} // namespace kernel
} // namespace trt_edgellm
