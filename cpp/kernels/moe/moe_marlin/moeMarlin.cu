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

/*
 * Edge-LLM MoE Marlin W4A16 GEMM Kernel Wrapper
 *
 * This file provides the rt::Tensor based interface for MoE W4A16 GEMM
 * using Marlin optimized kernels with AWQ (kU4) quantization format.
 *
 * AWQ format (kU4):
 * - Uses unsigned 4-bit integers with zero point = 8 baked in
 * - Dequantization: weight_fp16 = (weight_int4 - 8) * scale
 */

#include "moeMarlin.h"
#include <cuda_runtime.h>

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/stringUtils.h"
#include "marlin/scalar_type.hpp"

#include <algorithm>
#include <cstdint>

#include "marlin_moe_wna16/ops.cu"
#include "marlin_moe_wna16/sm80_kernel_bfloat16_fe2m1_bfloat16.cu"
#include "marlin_moe_wna16/sm80_kernel_float16_fe2m1_float16.cu"
#include "marlin_moe_wna16/sm80_kernel_float16_u4_float16.cu"

namespace trt_edgellm
{
namespace kernel
{

using namespace trt_edgellm::format;

// Internal constants for FP32 reduction (always enabled for accuracy)
constexpr bool kUseFp32Reduce = true;
constexpr bool kUseAtomicAdd = false;

void moeAwqW4A16MarlinGemm(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& weights,
    rt::Tensor const& scales, rt::Tensor const& sortedTokenIds, rt::Tensor const& expertIds,
    rt::Tensor const& numTokensPostPadded, rt::Tensor const& topkWeights, rt::Tensor& workspace, int64_t moeBlockSize,
    int64_t topK, bool mulTopkWeights, cudaStream_t stream)
{
    // Validate input shapes
    auto const inputShape = input.getShape();
    auto const outputShape = output.getShape();
    auto const weightsShape = weights.getShape();
    auto const scalesShape = scales.getShape();

    check::check(inputShape.getNumDims() == 2, "Input must be 2D tensor [numTokens, hiddenDim]");
    check::check(outputShape.getNumDims() == 2, "Output must be 2D tensor [numTokens*topK, outDim]");
    check::check(weightsShape.getNumDims() == 3, "Weights must be 3D tensor [numExperts, K/tile, N*tile/pack]");
    check::check(scalesShape.getNumDims() == 3, "Scales must be 3D tensor [numExperts, numGroups, outDim]");

    int64_t numTokens = inputShape[0];
    int64_t hiddenDim = inputShape[1];
    int64_t numExperts = weightsShape[0];
    int64_t numGroups = scalesShape[1];
    int64_t outDim = scalesShape[2];

    // Validate output shape
    // Note: For MoE, the input can be either:
    // 1. Original tokens [numTokens, K] → output [numTokens*topK, N] (first GEMM)
    // 2. Slot activations [numTokens*topK, K] → output [numTokens*topK, N] (second GEMM)
    // We accept both cases by checking if output matches input*topK OR just input
    bool validOutputShape = (outputShape[0] == numTokens * topK) || (outputShape[0] == numTokens);
    check::check(validOutputShape,
        fmtstr("Output shape[0] %ld must be either numTokens*topK=%ld or numTokens=%ld", outputShape[0],
            numTokens * topK, numTokens));
    check::check(outputShape[1] == outDim, fmtstr("Output shape[1] %ld != outDim %ld", outputShape[1], outDim));

    // Validate data types
    check::check(input.getDataType() == trt_edgellm::DataType::kHALF, "Input must be FP16");
    check::check(output.getDataType() == trt_edgellm::DataType::kHALF, "Output must be FP16");
    check::check(weights.getDataType() == trt_edgellm::DataType::kINT32, "Weights must be INT32 (Marlin packed)");
    check::check(scales.getDataType() == trt_edgellm::DataType::kHALF, "Scales must be FP16");
    check::check(sortedTokenIds.getDataType() == trt_edgellm::DataType::kINT32, "sortedTokenIds must be INT32");
    check::check(expertIds.getDataType() == trt_edgellm::DataType::kINT32, "expertIds must be INT32");
    check::check(
        numTokensPostPadded.getDataType() == trt_edgellm::DataType::kINT32, "numTokensPostPadded must be INT32");
    check::check(topkWeights.getDataType() == trt_edgellm::DataType::kFLOAT, "topkWeights must be FP32");
    check::check(workspace.getDataType() == trt_edgellm::DataType::kINT32, "workspace must be INT32");

    // Validate moe_block_size
    if (moeBlockSize != 8)
    {
        check::check(moeBlockSize % 16 == 0, fmtstr("moeBlockSize %ld must be divisible by 16", moeBlockSize));
        check::check(
            moeBlockSize >= 16 && moeBlockSize <= 64, fmtstr("moeBlockSize %ld must be in [16, 64]", moeBlockSize));
    }

    // Calculate group size
    int groupSize = (numGroups > 1) ? static_cast<int>(hiddenDim / numGroups) : -1;

    // Get device info
    int dev;
    int sms;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

    // AWQ W4A16 fixed scalar types
    trt_edgellm::marlin_dtypes::ScalarType aType = trt_edgellm::marlin_dtypes::kFloat16; // A16 activation
    trt_edgellm::marlin_dtypes::ScalarType bType
        = trt_edgellm::marlin_dtypes::kU4; // AWQ INT4 with zero point = 8 baked in
    trt_edgellm::marlin_dtypes::ScalarType cType = trt_edgellm::marlin_dtypes::kFloat16; // Output FP16
    trt_edgellm::marlin_dtypes::ScalarType sType = trt_edgellm::marlin_dtypes::kFloat16; // Scales FP16

    // Extract raw pointers
    void* inputPtr = const_cast<void*>(input.rawPointer());
    void* outputPtr = output.rawPointer();
    void* weightsPtr = const_cast<void*>(weights.rawPointer());
    void* scalesPtr = const_cast<void*>(scales.rawPointer());
    void* sortedTokenIdsPtr = const_cast<void*>(sortedTokenIds.rawPointer());
    void* expertIdsPtr = const_cast<void*>(expertIds.rawPointer());
    void* numTokensPostPaddedPtr = const_cast<void*>(numTokensPostPadded.rawPointer());
    void* topkWeightsPtr = const_cast<void*>(topkWeights.rawPointer());

    // Workspace layout: [locks (INT32)] [c_tmp (FP32)]
    int64_t numTokensPadded = sortedTokenIds.getShape()[0];
    int64_t locksSize = std::min(
        (outDim / marlin_moe_wna16::min_thread_n) * static_cast<int64_t>(divUp(numTokensPadded, moeBlockSize)),
        static_cast<int64_t>(sms * 4));

    // Round locksSize up to multiple of 4 to ensure cTmpPtr is 16-byte aligned
    // (since locks are INT32 = 4 bytes, we need 4 INT32s = 16 bytes alignment)
    locksSize = static_cast<int64_t>(divUp(locksSize, static_cast<int64_t>(4))) * 4;

    int32_t* workspaceBasePtr = static_cast<int32_t*>(workspace.rawPointer());
    void* locksPtr = workspaceBasePtr;

    // c_tmp buffer starts after locks (FP32 reduction buffer)
    void* cTmpPtr = workspaceBasePtr + locksSize;

    // Marlin uses locks for reductions; these must be zeroed each call.
    CUDA_CHECK(cudaMemsetAsync(locksPtr, 0, locksSize * sizeof(int32_t), stream));

    // Call the raw marlin_mm function with AWQ-specific parameters
    marlin_moe_wna16::marlin_mm(inputPtr, // A
        weightsPtr,                       // B
        outputPtr,                        // C
        cTmpPtr,                          // C_tmp (FP32 reduction buffer)
        nullptr,                          // b_bias (not used)
        nullptr,                          // a_scales (not used for A16)
        scalesPtr,                        // b_scales
        nullptr,                          // global_scale (not used)
        nullptr,                          // zp (not used - zero point = 8 baked in)
        nullptr,                          // g_idx (not used - no act_order)
        nullptr,                          // perm (not used - no act_order)
        nullptr,                          // a_tmp (not used - no act_order)
        sortedTokenIdsPtr, expertIdsPtr, numTokensPostPaddedPtr, topkWeightsPtr, static_cast<int>(moeBlockSize),
        static_cast<int>(numExperts), static_cast<int>(topK), mulTopkWeights,
        static_cast<int>(numTokens), // prob_m
        static_cast<int>(outDim),    // prob_n
        static_cast<int>(hiddenDim), // prob_k
        locksPtr, aType, bType, cType, sType,
        false, // has_bias
        false, // has_act_order
        true,  // is_k_full (always true for standard AWQ)
        false, // has_zp (zero point = 8 baked into dequant)
        static_cast<int>(numGroups), groupSize, dev, stream,
        -1, // thread_k (auto)
        -1, // thread_n (auto)
        sms,
        -1, // blocks_per_sm (auto)
        kUseAtomicAdd, kUseFp32Reduce,
        false // is_zp_float
    );
}

void moeNvfp4A16MarlinGemm(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& weights,
    rt::Tensor const& blockScales, rt::Tensor const& globalScales, rt::Tensor const& sortedTokenIds,
    rt::Tensor const& expertIds, rt::Tensor const& numTokensPostPadded, rt::Tensor const& topkWeights,
    rt::Tensor& workspace, int64_t moeBlockSize, int64_t topK, bool mulTopkWeights, cudaStream_t stream)
{
#if !SUPPORTS_FP8
    check::check(false, "NVFP4 A16 Marlin is unavailable: build does not support FP8 (SUPPORTS_FP8=0)");
#endif

    auto const inputShape = input.getShape();
    auto const outputShape = output.getShape();
    auto const weightsShape = weights.getShape();
    auto const scalesShape = blockScales.getShape();
    auto const globalScalesShape = globalScales.getShape();

    check::check(inputShape.getNumDims() == 2, "Input must be 2D [numTokens, hiddenDim]");
    check::check(outputShape.getNumDims() == 2, "Output must be 2D [numTokens*topK, outDim]");
    check::check(weightsShape.getNumDims() == 3, "Weights must be 3D [numExperts, K/16, 2*N]");
    check::check(scalesShape.getNumDims() == 3, "Block scales must be 3D [numExperts, K/16, N]");
    check::check(globalScalesShape.getNumDims() == 1, "Global scales must be 1D [numExperts]");

    int64_t const numTokens = inputShape[0];
    int64_t const hiddenDim = inputShape[1];
    int64_t const numExperts = weightsShape[0];
    int64_t const numGroups = scalesShape[1];
    int64_t const outDim = scalesShape[2];

    check::check(numTokens > 0 && hiddenDim > 0 && outDim > 0, "Input and output dimensions must be positive");
    check::check(topK > 0, "topK must be positive");
    check::check(hiddenDim % 16 == 0, fmtstr("Input K dimension %ld must be divisible by 16", hiddenDim));
    check::check(numGroups == hiddenDim / 16,
        fmtstr("Block-scale K groups %ld must equal hiddenDim/16=%ld", numGroups, hiddenDim / 16));
    check::check(weightsShape[1] == hiddenDim / 16,
        fmtstr("Packed weight K tiles %ld must equal hiddenDim/16=%ld", weightsShape[1], hiddenDim / 16));
    check::check(weightsShape[2] == 2 * outDim,
        fmtstr("Packed weight width %ld must equal 2*outDim=%ld", weightsShape[2], 2 * outDim));
    check::check(scalesShape[0] == numExperts, "Weight and block-scale expert dimensions must match");
    check::check(globalScalesShape[0] == numExperts, "Global-scale expert dimension must match weights");
    check::check(outputShape[0] == numTokens * topK,
        fmtstr("Output rows %ld must equal numTokens*topK=%ld", outputShape[0], numTokens * topK));
    check::check(
        outputShape[1] == outDim, fmtstr("Output width %ld must equal block-scale width %ld", outputShape[1], outDim));

    trt_edgellm::DataType const activationType = input.getDataType();
    check::check(activationType == trt_edgellm::DataType::kHALF || activationType == trt_edgellm::DataType::kBF16,
        "Input must be FP16 or BF16");
    check::check(output.getDataType() == activationType, "Output type must match input type");
    check::check(weights.getDataType() == trt_edgellm::DataType::kINT32,
        "Weights must be an INT32 view of Marlin-packed E2M1 codes");
    check::check(blockScales.getDataType() == trt_edgellm::DataType::kINT8,
        "Block scales must be raw E4M3 bytes exposed as INT8");
    check::check(globalScales.getDataType() == activationType, "Global-scale type must match input type");
    check::check(sortedTokenIds.getDataType() == trt_edgellm::DataType::kINT32, "sortedTokenIds must be INT32");
    check::check(expertIds.getDataType() == trt_edgellm::DataType::kINT32, "expertIds must be INT32");
    check::check(
        numTokensPostPadded.getDataType() == trt_edgellm::DataType::kINT32, "numTokensPostPadded must be INT32");
    check::check(topkWeights.getDataType() == trt_edgellm::DataType::kFLOAT, "topkWeights must be FP32");
    check::check(workspace.getDataType() == trt_edgellm::DataType::kINT32, "workspace must be INT32");

    if (moeBlockSize != 8)
    {
        check::check(moeBlockSize % 16 == 0, fmtstr("moeBlockSize %ld must be divisible by 16", moeBlockSize));
        check::check(
            moeBlockSize >= 16 && moeBlockSize <= 64, fmtstr("moeBlockSize %ld must be in [16, 64]", moeBlockSize));
    }

    int32_t dev = 0;
    int32_t sms = 0;
    int32_t major = 0;
    int32_t minor = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    check::check(major * 10 + minor >= 80, "NVFP4 A16 Marlin requires SM80 or newer");

    int64_t const numTokensPadded = sortedTokenIds.getShape()[0];
    int64_t locksSize = std::min(
        (outDim / marlin_moe_wna16::min_thread_n) * static_cast<int64_t>(divUp(numTokensPadded, moeBlockSize)),
        static_cast<int64_t>(sms * 4));
    locksSize = static_cast<int64_t>(divUp(locksSize, static_cast<int64_t>(4))) * 4;
    int64_t const requiredWorkspace = getMoeMarlinWorkspaceSize(numTokensPadded, outDim, moeBlockSize, sms);
    check::check(workspace.getShape().volume() >= requiredWorkspace,
        fmtstr("Marlin workspace has %ld elements but requires at least %ld", workspace.getShape().volume(),
            requiredWorkspace));

    int32_t* workspaceBase = static_cast<int32_t*>(workspace.rawPointer());
    void* locks = workspaceBase;
    void* cTmp = workspaceBase + locksSize;
    CUDA_CHECK(cudaMemsetAsync(locks, 0, locksSize * sizeof(int32_t), stream));

    trt_edgellm::marlin_dtypes::ScalarType const activationScalarType = activationType == trt_edgellm::DataType::kHALF
        ? trt_edgellm::marlin_dtypes::kFloat16
        : trt_edgellm::marlin_dtypes::kBFloat16;

    marlin_moe_wna16::marlin_mm(
        const_cast<void*>(input.rawPointer()), const_cast<void*>(weights.rawPointer()), output.rawPointer(), cTmp,
        nullptr, // b_bias
        nullptr, // a_scales
        const_cast<void*>(blockScales.rawPointer()), const_cast<void*>(globalScales.rawPointer()),
        nullptr, // zero points
        nullptr, // g_idx
        nullptr, // perm
        nullptr, // activation permutation workspace
        const_cast<void*>(sortedTokenIds.rawPointer()), const_cast<void*>(expertIds.rawPointer()),
        const_cast<void*>(numTokensPostPadded.rawPointer()), const_cast<void*>(topkWeights.rawPointer()),
        static_cast<int>(moeBlockSize), static_cast<int>(numExperts), static_cast<int>(topK), mulTopkWeights,
        static_cast<int>(numTokens), static_cast<int>(outDim), static_cast<int>(hiddenDim), locks, activationScalarType,
        trt_edgellm::marlin_dtypes::kFE2M1f, activationScalarType, trt_edgellm::marlin_dtypes::kFE4M3fn,
        false, // has_bias
        false, // has_act_order
        true,  // is_k_full
        false, // has_zp
        static_cast<int>(numGroups), 16, dev, stream,
        -1, // thread_k (auto)
        -1, // thread_n (auto)
        sms,
        -1, // blocks_per_sm (auto)
        kUseAtomicAdd, kUseFp32Reduce,
        false // is_zp_float
    );
}

int64_t getMoeMarlinWorkspaceSize(int64_t numTokensPadded, int64_t outDim, int64_t moeBlockSize, int64_t numSMs)
{
    // Locks size (INT32)
    int64_t maxNTiles = outDim / marlin_moe_wna16::min_thread_n;
    int64_t numBlocks = static_cast<int64_t>(divUp(numTokensPadded, moeBlockSize));
    int64_t locksSize = std::min(maxNTiles * numBlocks, numSMs * 4);
    // Round locksSize up to multiple of 4 to ensure cTmpPtr is 16-byte aligned.
    locksSize = divUp(locksSize, 4) * 4;

    // C_tmp size for FP32 reduction (same element size as INT32)
    int64_t cTmpSize = std::min(
        outDim * numTokensPadded, numSMs * 4 * moeBlockSize * static_cast<int64_t>(marlin_moe_wna16::max_thread_n));
    if (moeBlockSize == 8)
    {
        cTmpSize *= 2;
    }

    // Total workspace = locks + c_tmp (both are 4 bytes per element)
    return locksSize + cTmpSize;
}

} // namespace kernel
} // namespace trt_edgellm
