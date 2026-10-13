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

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED

#include "cuteDslNvfp4MoeSm110Runner.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "kernels/moe/NvFP4MoEUtils.h"
#include "kernels/moe/fp4SupportKernels/buildLayout.h"
#include "kernels/moe/fp4SupportKernels/fp4Quantize.h"
#include "kernels/moe/nvfp4_cutedsl/nvfp4MoePdlConfig.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>

namespace trt_edgellm
{

detail::LazyKernelModule<nvfp4_moe_sm110_fc1_relu2_n128_Kernel_Module_t> CuteDslNvfp4MoeSm110Runner::sFC1Relu2N128{};
detail::LazyKernelModule<nvfp4_moe_sm110_fc1_swiglu_n128_Kernel_Module_t> CuteDslNvfp4MoeSm110Runner::sFC1SwiGLUN128{};
detail::LazyKernelModule<nvfp4_moe_sm110_fc1_geglu_n128_Kernel_Module_t> CuteDslNvfp4MoeSm110Runner::sFC1GeGLUN128{};
detail::LazyKernelModule<nvfp4_moe_sm110_fc2_n128_fp16_Kernel_Module_t> CuteDslNvfp4MoeSm110Runner::sFC2N128Fp16{};

namespace
{
constexpr int32_t kACT_SWIGLU{2};
constexpr int32_t kACT_RELU2{4};
constexpr int32_t kACT_GEGLU{5};
constexpr int32_t kIODT_FP16{1};
constexpr size_t kDeviceAlignment{256};

inline size_t alignUp(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

inline int64_t padUp64(int64_t value, int64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

inline bool isSupportedSm(int32_t smVersion)
{
    return smVersion == 100 || smVersion == 101 || smVersion == 110;
}

inline int64_t maxPermutedRows(int64_t routedRows, int64_t numExperts)
{
    return padUp64(routedRows + numExperts * (CuteDslNvfp4MoeSm110Runner::kRowTileAlign - 1),
        CuteDslNvfp4MoeSm110Runner::kRowTileAlign);
}

struct Sm110WorkspaceLayout
{
    size_t inputFP4{};
    size_t inputSF{};
    size_t tileGroup{};
    size_t tileLimit{};
    size_t permutedToExpanded{};
    size_t numTiles{};
    size_t fc1FP4{};
    size_t fc1SF{};
    size_t total{};

    int32_t permutedM{};
    int32_t numTileEntries{};
    int32_t fc1InputN{};
    int32_t paddedSfColsH{};
    int32_t paddedSfColsI{};
};

Sm110WorkspaceLayout buildWorkspaceLayout(int32_t numTokens, int32_t routedRows, int32_t numExperts, int32_t hiddenSize,
    int32_t moeInterSize, int32_t activationType)
{
    Sm110WorkspaceLayout L{};
    int64_t const routedRowsForCap = std::max<int64_t>(1, routedRows);
    L.permutedM = static_cast<int32_t>(maxPermutedRows(routedRowsForCap, numExperts));
    L.numTileEntries = std::max(1, L.permutedM / CuteDslNvfp4MoeSm110Runner::kRowTileAlign);
    L.fc1InputN = (activationType == kACT_SWIGLU || activationType == kACT_GEGLU) ? 2 * moeInterSize : moeInterSize;
    L.paddedSfColsH = static_cast<int32_t>(padUp64(hiddenSize / CuteDslNvfp4MoeSm110Runner::kNvfp4SfVecSize, 4));
    L.paddedSfColsI = static_cast<int32_t>(padUp64(moeInterSize / CuteDslNvfp4MoeSm110Runner::kNvfp4SfVecSize, 4));

    size_t cursor = 0;
    auto place = [&cursor](size_t& offset, size_t bytes) {
        offset = cursor;
        cursor = alignUp(cursor + bytes, kDeviceAlignment);
    };

    (void) numTokens;
    int64_t const inputRows = std::max<int64_t>(1, routedRowsForCap);
    place(L.inputFP4, static_cast<size_t>(inputRows) * static_cast<size_t>(hiddenSize / 2));
    place(L.inputSF,
        static_cast<size_t>(inputRows) * static_cast<size_t>(hiddenSize / CuteDslNvfp4MoeSm110Runner::kNvfp4SfVecSize));
    place(L.tileGroup, static_cast<size_t>(L.numTileEntries) * sizeof(int32_t));
    place(L.tileLimit, static_cast<size_t>(L.numTileEntries) * sizeof(int32_t));
    place(L.permutedToExpanded, static_cast<size_t>(L.permutedM) * sizeof(int32_t));
    place(L.numTiles, sizeof(int32_t));
    place(L.fc1FP4, static_cast<size_t>(L.permutedM) * static_cast<size_t>(moeInterSize / 2));
    place(L.fc1SF, static_cast<size_t>(L.permutedM) * static_cast<size_t>(L.paddedSfColsI));
    L.total = cursor + kDeviceAlignment;
    return L;
}

inline void* offsetPtr(void* base, size_t offset)
{
    return static_cast<void*>(static_cast<std::byte*>(base) + offset);
}

bool useFastDecodeSetup(CuteDslNvfp4MoeSm110Params const& params)
{
    // numTokens == 1 selects the fused fp4BuildLayoutAndQuantizeRoutedLinearSFDecode
    // setup over the general (buildLayoutGpu + fp4QuantizeRoutedLinearSF) path: the
    // fused kernel's host launcher rejects M != 1. Its remaining bounds are already
    // guaranteed by canImplement() (topK <= kMaxTopK = 8) and the supported expert
    // set {128, 256}, which fp4Quantize.cu's kMaxDecodeExperts (256) covers.
    return params.numTokens == 1;
}

} // namespace

bool CuteDslNvfp4MoeSm110Runner::canImplement(int32_t hiddenSize, int32_t moeInterSize, int32_t numExperts,
    int32_t topK, int32_t smVersion, int32_t activationType, int32_t ioDtype, int32_t backend)
{
    (void) backend;
    if (!isSupportedSm(smVersion) || ioDtype != kIODT_FP16)
    {
        return false;
    }
    if (activationType != kACT_SWIGLU && activationType != kACT_RELU2 && activationType != kACT_GEGLU)
    {
        return false;
    }
    // The FC1/FC2 cubins are runtime-polymorphic in L (num_experts); the runner
    // restricts E to the product-supported set {128, 256} (see kSupportedNumExperts).
    if (!isSupportedNumExperts(numExperts) || topK <= 0 || topK > kMaxTopK)
    {
        return false;
    }
    if (hiddenSize <= 0 || hiddenSize % kHiddenSizeAlignment != 0)
    {
        return false;
    }
    int32_t const fc1InputN
        = (activationType == kACT_SWIGLU || activationType == kACT_GEGLU) ? 2 * moeInterSize : moeInterSize;
    if (moeInterSize <= 0 || moeInterSize % 64 != 0 || fc1InputN % kLevelTileN != 0)
    {
        return false;
    }
    return true;
}

bool CuteDslNvfp4MoeSm110Runner::ensureKernelModules(CuteDslNvfp4MoeSm110Params const& params, cudaStream_t stream)
{
    bool fc1Loaded{false};
    if (params.activationType == kACT_RELU2)
    {
        fc1Loaded = detail::ensureModuleLoaded<nvfp4_moe_sm110_fc1_relu2_n128_Kernel_Module_Load,
            nvfp4_moe_sm110_fc1_relu2_n128_Kernel_Module_Unload>(
            sFC1Relu2N128, "nvfp4_moe_sm110_fc1_relu2_n128", stream);
    }
    else if (params.activationType == kACT_SWIGLU)
    {
        fc1Loaded = detail::ensureModuleLoaded<nvfp4_moe_sm110_fc1_swiglu_n128_Kernel_Module_Load,
            nvfp4_moe_sm110_fc1_swiglu_n128_Kernel_Module_Unload>(
            sFC1SwiGLUN128, "nvfp4_moe_sm110_fc1_swiglu_n128", stream);
    }
    else if (params.activationType == kACT_GEGLU)
    {
        fc1Loaded = detail::ensureModuleLoaded<nvfp4_moe_sm110_fc1_geglu_n128_Kernel_Module_Load,
            nvfp4_moe_sm110_fc1_geglu_n128_Kernel_Module_Unload>(
            sFC1GeGLUN128, "nvfp4_moe_sm110_fc1_geglu_n128", stream);
    }
    else
    {
        LOG_ERROR("CuteDslNvfp4MoeSm110Runner: unsupported activation_type=%d", params.activationType);
        return false;
    }
    return fc1Loaded
        && detail::ensureModuleLoaded<nvfp4_moe_sm110_fc2_n128_fp16_Kernel_Module_Load,
            nvfp4_moe_sm110_fc2_n128_fp16_Kernel_Module_Unload>(sFC2N128Fp16, "nvfp4_moe_sm110_fc2_n128_fp16", stream);
}

size_t CuteDslNvfp4MoeSm110Runner::getWorkspaceSize(int32_t maxNumTokens, int32_t maxRoutedRows, int32_t numExperts,
    int32_t topK, int32_t hiddenSize, int32_t moeInterSize)
{
    if (maxNumTokens <= 0 || maxRoutedRows <= 0 || numExperts <= 0 || topK <= 0 || hiddenSize <= 0 || moeInterSize <= 0)
    {
        return 0;
    }
    int32_t const routedRows = static_cast<int32_t>(
        std::min<int64_t>(std::max<int64_t>(static_cast<int64_t>(maxNumTokens) * topK, maxRoutedRows),
            static_cast<int64_t>(std::numeric_limits<int32_t>::max())));
    auto const swigluLayout
        = buildWorkspaceLayout(maxNumTokens, routedRows, numExperts, hiddenSize, moeInterSize, kACT_SWIGLU);
    auto const relu2Layout
        = buildWorkspaceLayout(maxNumTokens, routedRows, numExperts, hiddenSize, moeInterSize, kACT_RELU2);
    return std::max(swigluLayout.total, relu2Layout.total);
}

int32_t CuteDslNvfp4MoeSm110Runner::run(CuteDslNvfp4MoeSm110Params const& params, void* workspace, cudaStream_t stream)
{
    if (!ensureKernelModules(params, stream))
    {
        return -1;
    }
    if (workspace == nullptr)
    {
        LOG_ERROR("CuteDslNvfp4MoeSm110Runner: null workspace pointer");
        return -1;
    }
    int32_t const routedRows = params.numTokens * params.topK;
    auto const L = buildWorkspaceLayout(
        params.numTokens, routedRows, params.numExperts, params.hiddenSize, params.moeInterSize, params.activationType);

    void* const inputFP4 = offsetPtr(workspace, L.inputFP4);
    void* const inputSF = offsetPtr(workspace, L.inputSF);
    void* const tileGroup = offsetPtr(workspace, L.tileGroup);
    void* const tileLimit = offsetPtr(workspace, L.tileLimit);
    void* const permutedToExpanded = offsetPtr(workspace, L.permutedToExpanded);
    void* const numTiles = offsetPtr(workspace, L.numTiles);
    void* const fc1FP4 = offsetPtr(workspace, L.fc1FP4);
    void* const fc1SF = offsetPtr(workspace, L.fc1SF);

    kernel::MoELayoutBuffers layoutBuffers{};
    layoutBuffers.tileIdxToGroupIdx
        = rt::Tensor(tileGroup, {L.numTileEntries}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    layoutBuffers.tileIdxToMnLimit
        = rt::Tensor(tileLimit, {L.numTileEntries}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    layoutBuffers.permutedIdxToExpandedIdx
        = rt::Tensor(permutedToExpanded, {L.permutedM}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    layoutBuffers.numNonExitingTiles = rt::Tensor(numTiles, {1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor hiddenT(const_cast<void*>(params.hiddenStates), {params.numTokens, params.hiddenSize},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor topkIdsT(const_cast<int32_t*>(params.topkIds), {params.numTokens, params.topK}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kINT32);
    rt::Tensor inputScaleT(const_cast<float*>(params.inputGlobalScale), {params.numExperts}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT);
    rt::Tensor inputFP4T(
        inputFP4, {routedRows, params.hiddenSize / 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);
    rt::Tensor inputSFT(
        inputSF, {routedRows, params.hiddenSize / kNvfp4SfVecSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8);

    // FC1 clears output after its dependency wait. FC2's existing dependent
    // wait remains the completion and visibility boundary.
    int64_t const outputElements = static_cast<int64_t>(params.numTokens) * params.hiddenSize;

    if (useFastDecodeSetup(params))
    {
        kernel::fp4BuildLayoutAndQuantizeRoutedLinearSFDecode(hiddenT, topkIdsT, inputScaleT, layoutBuffers, inputFP4T,
            inputSFT, params.numExperts, kRowTileAlign, stream);
    }
    else
    {
        kernel::buildLayoutGpu(
            layoutBuffers, params.topkIds, params.numTokens, params.topK, params.numExperts, kRowTileAlign, stream);
        CUDA_CHECK(cudaGetLastError());
        kernel::fp4QuantizeRoutedLinearSF(hiddenT, topkIdsT, inputScaleT, inputFP4T, inputSFT, stream);
    }
    CUDA_CHECK(cudaGetLastError());

    int32_t ret = -1;
    int64_t const origM = routedRows;
    int64_t const m = L.permutedM;
    int64_t const n1 = L.fc1InputN;
    int64_t const h = params.hiddenSize;
    int64_t const e = params.numExperts;
    int32_t const enablePdl = useNvfp4MoePdl(params.enablePdl, getSMVersion()) ? 1 : 0;

    if (params.activationType == kACT_RELU2)
    {
        ret = cute_dsl_nvfp4_moe_sm110_fc1_relu2_n128_wrapper(&sFC1Relu2N128.module, inputFP4,
            const_cast<void*>(params.fc1QWeights), inputSF, const_cast<void*>(params.fc1BlocksScale), fc1FP4, fc1SF,
            params.output, outputElements, /*fuse_output_zero=*/1,
            const_cast<void*>(static_cast<void const*>(params.fc1Alpha)),
            const_cast<void*>(static_cast<void const*>(params.inputGlobalScale)),
            const_cast<void*>(static_cast<void const*>(params.downInputScale)), tileGroup, tileLimit,
            permutedToExpanded, numTiles, origM, m, n1, h, e, getDeviceMultiProcessorCount(), enablePdl, stream);
    }
    else if (params.activationType == kACT_SWIGLU)
    {
        ret = cute_dsl_nvfp4_moe_sm110_fc1_swiglu_n128_wrapper(&sFC1SwiGLUN128.module, inputFP4,
            const_cast<void*>(params.fc1QWeights), inputSF, const_cast<void*>(params.fc1BlocksScale), fc1FP4, fc1SF,
            params.output, outputElements, /*fuse_output_zero=*/1,
            const_cast<void*>(static_cast<void const*>(params.fc1Alpha)),
            const_cast<void*>(static_cast<void const*>(params.inputGlobalScale)),
            const_cast<void*>(static_cast<void const*>(params.downInputScale)), tileGroup, tileLimit,
            permutedToExpanded, numTiles, origM, m, n1, h, e, getDeviceMultiProcessorCount(), enablePdl, stream);
    }
    else if (params.activationType == kACT_GEGLU)
    {
        ret = cute_dsl_nvfp4_moe_sm110_fc1_geglu_n128_wrapper(&sFC1GeGLUN128.module, inputFP4,
            const_cast<void*>(params.fc1QWeights), inputSF, const_cast<void*>(params.fc1BlocksScale), fc1FP4, fc1SF,
            params.output, outputElements, /*fuse_output_zero=*/1,
            const_cast<void*>(static_cast<void const*>(params.fc1Alpha)),
            const_cast<void*>(static_cast<void const*>(params.inputGlobalScale)),
            const_cast<void*>(static_cast<void const*>(params.downInputScale)), tileGroup, tileLimit,
            permutedToExpanded, numTiles, origM, m, n1, h, e, getDeviceMultiProcessorCount(), enablePdl, stream);
    }
    else
    {
        LOG_ERROR("CuteDslNvfp4MoeSm110Runner: unsupported activation_type=%d", params.activationType);
        return -1;
    }
    if (ret != 0)
    {
        LOG_ERROR("CuteDslNvfp4MoeSm110Runner: FC1 kernel returned error code %d", ret);
        return ret;
    }
    CUDA_CHECK(cudaGetLastError());

    ret = cute_dsl_nvfp4_moe_sm110_fc2_n128_fp16_wrapper(&sFC2N128Fp16.module, fc1FP4,
        const_cast<void*>(params.fc2QWeights), fc1SF, const_cast<void*>(params.fc2BlocksScale), params.output,
        const_cast<void*>(static_cast<void const*>(params.fc2Alpha)),
        const_cast<void*>(static_cast<void const*>(params.downInputScale)), tileGroup, tileLimit, permutedToExpanded,
        numTiles, const_cast<void*>(static_cast<void const*>(params.topkWeights)), m, h, params.moeInterSize, e,
        params.numTokens, params.topK, getDeviceMultiProcessorCount(), enablePdl, stream);
    if (ret != 0)
    {
        LOG_ERROR("CuteDslNvfp4MoeSm110Runner: FC2 kernel returned error code %d", ret);
        return ret;
    }
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

} // namespace trt_edgellm

#endif // CUTE_DSL_NVFP4_MOE_ENABLED
