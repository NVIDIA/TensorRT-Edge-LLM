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

#include "fusedNvfp4GemmAllReducePlugin.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "plugins/allReducePlugin/allReducePlugin.h"
#include "plugins/utils/pluginUtils.h"

#include "kernels/multiDeviceKernels/scaleConvert.h"

#include "kernels/multiDeviceKernels/cuteDslGemmNvFp4Runner.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kFUSED_NVFP4_GEMM_AR_PLUGIN_VERSION{"1"};
constexpr char const* kFUSED_NVFP4_GEMM_AR_PLUGIN_NAME{"FusedNvfp4GemmAllReducePlugin"};

// FP4 inputs: [fp4_act, act_scale, weight_f4, weight_f8_scale, weight_f8_scale_f32_scale]
constexpr int32_t kFP4_ACT_IDX{0};
constexpr int32_t kFP4_ACT_SCALE_IDX{1};
constexpr int32_t kFP4_WEIGHT_IDX{2};
constexpr int32_t kFP4_WEIGHT_F8_SCALE_IDX{3};
constexpr int32_t kFP4_WEIGHT_F32_SCALE_IDX{4};
constexpr int32_t kFP4_NUM_INPUTS{5};
constexpr int32_t kNUM_OUTPUTS{1};
constexpr int32_t kOUT_TENSOR_IDX{0};

using NcclAllReduceFn = int (*)(void const*, void*, size_t, int, int, void*, cudaStream_t);

constexpr int32_t kNcclFloat16 = 6;
constexpr int32_t kNcclSum = 0;
constexpr int32_t kNcclSuccess = 0;

} // namespace

PluginFieldCollection FusedNvfp4GemmAllReducePluginCreator::mFieldCollection{};
std::vector<PluginField> FusedNvfp4GemmAllReducePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedNvfp4GemmAllReducePluginCreator);

// ========================== Plugin Implementation ==========================

FusedNvfp4GemmAllReducePlugin::FusedNvfp4GemmAllReducePlugin(std::string const& name, int32_t tpSize)
    : mLayerName(name)
    , mTpSize(tpSize)
{
    LOG_DEBUG("FusedNvfp4GemmAllReducePlugin created: name=%s, tpSize=%d", name.c_str(), tpSize);
}

FusedNvfp4GemmAllReducePlugin::FusedNvfp4GemmAllReducePlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    if (fc == nullptr)
    {
        throw std::invalid_argument("FusedNvfp4GemmAllReducePlugin requires plugin fields");
    }
    auto tpSize = parsePluginScalarField<int32_t>("tp_size", fc);
    if (!tpSize.has_value())
    {
        throw std::invalid_argument("FusedNvfp4GemmAllReducePlugin requires 'tp_size' field");
    }
    mTpSize = tpSize.value();
    LOG_DEBUG("FusedNvfp4GemmAllReducePlugin deserialized from fields: name=%s, tpSize=%d", name.c_str(), mTpSize);
}

FusedNvfp4GemmAllReducePlugin::~FusedNvfp4GemmAllReducePlugin()
{
    delete mGemmRunner;
    mGemmRunner = nullptr;
    if (mCachedWeightScaleTiled)
    {
        cudaFree(mCachedWeightScaleTiled);
        mCachedWeightScaleTiled = nullptr;
        mCachedWeightScaleTiledSize = 0;
        mCachedWeightNumRows = 0;
        mCachedWeightNumKBlocks = 0;
    }
}

IPluginCapability* FusedNvfp4GemmAllReducePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

IPluginV3* FusedNvfp4GemmAllReducePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new FusedNvfp4GemmAllReducePlugin(mLayerName, mTpSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        // Runtime-owned CUDA resources are rebuilt lazily per clone.
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("FusedNvfp4GemmAllReducePlugin clone failed: %s", e.what());
        return nullptr;
    }
}

int32_t FusedNvfp4GemmAllReducePlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

int32_t FusedNvfp4GemmAllReducePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* /* inputTypes */, int32_t nbInputs) const noexcept
{
    try
    {
        if (nbInputs != 5 || nbOutputs != 1)
        {
            return -1;
        }
        outputTypes[kOUT_TENSOR_IDX] = DataType::kHALF;
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t FusedNvfp4GemmAllReducePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& /* exprBuilder */) noexcept
{
    try
    {
        if (nbInputs != 5 || nbOutputs != 1)
        {
            return -1;
        }
        outputs[kOUT_TENSOR_IDX].nbDims = inputs[kFP4_ACT_IDX].nbDims;
        for (int32_t d = 0; d < outputs[kOUT_TENSOR_IDX].nbDims - 1; ++d)
        {
            outputs[kOUT_TENSOR_IDX].d[d] = inputs[kFP4_ACT_IDX].d[d];
        }
        outputs[kOUT_TENSOR_IDX].d[outputs[kOUT_TENSOR_IDX].nbDims - 1] = inputs[kFP4_WEIGHT_IDX].d[0];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool FusedNvfp4GemmAllReducePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != kFP4_NUM_INPUTS || nbOutputs != kNUM_OUTPUTS || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    auto const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    bool const isOutput = (pos >= nbInputs);
    if (isOutput)
    {
        return desc.type == DataType::kHALF;
    }
    if (pos == kFP4_ACT_IDX)
    {
#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 8)
        return desc.type == DataType::kFP4 || desc.type == DataType::kINT8 || desc.type == DataType::kFP8;
#else
        return desc.type == DataType::kINT8 || desc.type == DataType::kFP8;
#endif
    }
    if (pos == kFP4_WEIGHT_IDX)
    {
#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 8)
        return desc.type == DataType::kFP4 || desc.type == DataType::kINT8 || desc.type == DataType::kFP8;
#else
        return desc.type == DataType::kINT8 || desc.type == DataType::kFP8;
#endif
    }
    if (pos == kFP4_ACT_SCALE_IDX)
    {
        return desc.type == DataType::kFLOAT;
    }
    if (pos == kFP4_WEIGHT_F8_SCALE_IDX)
    {
        return desc.type == DataType::kFP8 || desc.type == DataType::kINT8 || desc.type == DataType::kFLOAT;
    }
    if (pos == kFP4_WEIGHT_F32_SCALE_IDX)
    {
        return desc.type == DataType::kFLOAT;
    }
    return true;
}

int32_t FusedNvfp4GemmAllReducePlugin::configurePlugin(DynamicPluginTensorDesc const* /* in */, int32_t nbInputs,
    DynamicPluginTensorDesc const* /* out */, int32_t nbOutputs) noexcept
{
    return (nbInputs == 5 && nbOutputs == 1) ? 0 : -1;
}

size_t FusedNvfp4GemmAllReducePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    try
    {
        auto const& actScaleDims = inputs[kFP4_ACT_SCALE_IDX].max;
        int32_t actScaleNumKBlocks = actScaleDims.d[actScaleDims.nbDims - 1];
        int64_t actScaleElements = 1;
        for (int d = 0; d < actScaleDims.nbDims; ++d)
        {
            actScaleElements *= actScaleDims.d[d];
        }
        int32_t actScaleNumRows = static_cast<int32_t>(actScaleElements / actScaleNumKBlocks);

        auto const& wScaleDims = inputs[kFP4_WEIGHT_F8_SCALE_IDX].max;
        int32_t wScaleNumKBlocks = wScaleDims.d[wScaleDims.nbDims - 1];
        int64_t wScaleElements = 1;
        for (int d = 0; d < wScaleDims.nbDims; ++d)
        {
            wScaleElements *= wScaleDims.d[d];
        }
        int32_t wScaleNumRows = static_cast<int32_t>(wScaleElements / wScaleNumKBlocks);

        // Workspace layout:
        //   [0] SfAtom tiled activation scale
        //   [1] SfAtom tiled weight scale (for graph capture when cache is not populated)
        size_t const actTiled
            = static_cast<size_t>(kernels::getSfAtomTiledBufferSize(actScaleNumRows, actScaleNumKBlocks));
        size_t const weightTiled
            = static_cast<size_t>(kernels::getSfAtomTiledBufferSize(wScaleNumRows, wScaleNumKBlocks));

        size_t workspaceSize = 0;
        workspaceSize
            = accumulateWorkspaceSize(workspaceSize, rt::Coords{static_cast<int64_t>(actTiled)}, DataType::kINT8);
        workspaceSize
            = accumulateWorkspaceSize(workspaceSize, rt::Coords{static_cast<int64_t>(weightTiled)}, DataType::kINT8);
        return workspaceSize;
    }
    catch (std::exception const& e)
    {
        return 0;
    }
}

int32_t FusedNvfp4GemmAllReducePlugin::enqueue(PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    try
    {
#if !SUPPORTS_FP8
        LOG_ERROR("FusedNvfp4GemmAllReducePlugin requires CUDA_VERSION >= 11080 (cuda_fp8.h unavailable).");
        return -1;
#else
        if (!ensureGemmRunners())
        {
            return -1;
        }

        int32_t const actIdx = kFP4_ACT_IDX;
        int32_t const wIdx = kFP4_WEIGHT_IDX;
        auto const& actDesc = inputDesc[actIdx];
        auto const& wDesc = inputDesc[wIdx];

        if (actDesc.dims.nbDims <= 0 || wDesc.dims.nbDims <= 0)
        {
            LOG_ERROR("FusedNvfp4GemmAllReducePlugin: invalid input ranks (act=%d, weight=%d)", actDesc.dims.nbDims,
                wDesc.dims.nbDims);
            return -1;
        }
        // TensorRT reports FP4 tensors with logical element dimensions. The physical
        // FP4 storage is packed, but the CuTe DSL runner API takes logical K and
        // handles packed A/B pointers internally.
        int32_t const K = actDesc.dims.d[actDesc.dims.nbDims - 1];
        int32_t const N = wDesc.dims.d[0];
        int64_t M = 1;
        for (int32_t d = 0; d < actDesc.dims.nbDims - 1; ++d)
        {
            M *= actDesc.dims.d[d];
        }
        if (M <= 0 || N <= 0 || K <= 0)
        {
            LOG_ERROR(
                "FusedNvfp4GemmAllReducePlugin: invalid GEMM shape M=%lld N=%d K=%d", static_cast<long long>(M), N, K);
            return -1;
        }
        if (M > std::numeric_limits<int32_t>::max())
        {
            LOG_ERROR("FusedNvfp4GemmAllReducePlugin: M=%lld exceeds int32 capacity", static_cast<long long>(M));
            return -1;
        }
        int32_t const m = static_cast<int32_t>(M);

        int currentDevice = -1;
        CUDA_CHECK(cudaGetDevice(&currentDevice));

        int64_t const outElements = M * N;

        // Cache weight scale UE4M3 + SfAtom tiled layout when the stream is not
        // being captured. During CUDA graph capture, use the TRT workspace path.
        if (mCachedWeightScaleTiled == nullptr)
        {
            cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
            CUDA_CHECK(cudaStreamIsCapturing(stream, &captureStatus));

            // cudaMalloc/cache population is not capture-safe. When TensorRT is
            // capturing the enqueue stream, use the TensorRT-provided workspace
            // for this call instead of mutating the persistent cache.
            if (captureStatus != cudaStreamCaptureStatusActive)
            {
                auto const& wScaleDims = inputDesc[kFP4_WEIGHT_F8_SCALE_IDX].dims;
                int32_t wNumKBlocks = wScaleDims.d[wScaleDims.nbDims - 1];
                int64_t wScaleElements = 1;
                for (int32_t d = 0; d < wScaleDims.nbDims; ++d)
                {
                    wScaleElements *= wScaleDims.d[d];
                }
                int32_t wNumRows = static_cast<int32_t>(wScaleElements / wNumKBlocks);

                mCachedWeightScaleTiledSize = kernels::getSfAtomTiledBufferSize(wNumRows, wNumKBlocks);
                cudaError_t const allocErr = cudaMalloc(&mCachedWeightScaleTiled, mCachedWeightScaleTiledSize);
                if (allocErr != cudaSuccess || mCachedWeightScaleTiled == nullptr)
                {
                    LOG_ERROR(
                        "FusedNvfp4GemmAllReducePlugin: cudaMalloc for weight scale cache failed (%s, %ld bytes); "
                        "using workspace path",
                        cudaGetErrorString(allocErr), static_cast<long>(mCachedWeightScaleTiledSize));
                    mCachedWeightScaleTiled = nullptr;
                    mCachedWeightScaleTiledSize = 0;
                }
                else
                {
                    kernels::fusedFp8ToSfAtom(static_cast<__nv_fp8_e4m3 const*>(inputs[kFP4_WEIGHT_F8_SCALE_IDX]),
                        static_cast<float const*>(inputs[kFP4_WEIGHT_F32_SCALE_IDX]), mCachedWeightScaleTiled, wNumRows,
                        wNumKBlocks, stream);

                    mCachedWeightNumRows = wNumRows;
                    mCachedWeightNumKBlocks = wNumKBlocks;

                    LOG_DEBUG(
                        "FusedNvfp4GemmAllReducePlugin: Cached weight scale (rows=%d, kBlocks=%d, tiledSize=%ld bytes)",
                        wNumRows, wNumKBlocks, static_cast<long>(mCachedWeightScaleTiledSize));
                }
            }
        }

        std::byte* workspaceCursor = static_cast<std::byte*>(workspace);
        uint8_t* globalActScaleTiled = nullptr;
        int64_t globalActTiledSize = 0;
        {
            auto const& actScaleDims = inputDesc[kFP4_ACT_SCALE_IDX].dims;
            int32_t actNumKBlocks = actScaleDims.d[actScaleDims.nbDims - 1];
            int64_t actScaleElements = 1;
            for (int32_t d = 0; d < actScaleDims.nbDims; ++d)
            {
                actScaleElements *= actScaleDims.d[d];
            }
            int32_t actNumRows = static_cast<int32_t>(actScaleElements / actNumKBlocks);

            globalActTiledSize = kernels::getSfAtomTiledBufferSize(actNumRows, actNumKBlocks);
            rt::Tensor globalActScaleTiledTensor
                = assignTensorFromWorkspace(workspaceCursor, rt::Coords{globalActTiledSize}, DataType::kINT8);
            globalActScaleTiled = static_cast<uint8_t*>(globalActScaleTiledTensor.rawPointer());
            kernels::fusedFp32ToSfAtom(static_cast<float const*>(inputs[kFP4_ACT_SCALE_IDX]), globalActScaleTiled,
                actNumRows, actNumKBlocks, stream);
        }

        uint8_t const* weightSFB = mCachedWeightScaleTiled;
        if (weightSFB == nullptr)
        {
            auto const& wScaleDims = inputDesc[kFP4_WEIGHT_F8_SCALE_IDX].dims;
            int32_t wNumKBlocks = wScaleDims.d[wScaleDims.nbDims - 1];
            int64_t wScaleElements = 1;
            for (int32_t d = 0; d < wScaleDims.nbDims; ++d)
            {
                wScaleElements *= wScaleDims.d[d];
            }
            int32_t wNumRows = static_cast<int32_t>(wScaleElements / wNumKBlocks);
            int64_t const weightTiledSize = kernels::getSfAtomTiledBufferSize(wNumRows, wNumKBlocks);
            rt::Tensor weightScaleTiledTensor
                = assignTensorFromWorkspace(workspaceCursor, rt::Coords{weightTiledSize}, DataType::kINT8);
            auto* weightScaleTiled = static_cast<uint8_t*>(weightScaleTiledTensor.rawPointer());
            kernels::fusedFp8ToSfAtom(static_cast<__nv_fp8_e4m3 const*>(inputs[kFP4_WEIGHT_F8_SCALE_IDX]),
                static_cast<float const*>(inputs[kFP4_WEIGHT_F32_SCALE_IDX]), weightScaleTiled, wNumRows, wNumKBlocks,
                stream);
            weightSFB = weightScaleTiled;
        }

        static std::atomic<bool> sNcclPathLogged{false};
        if (!sNcclPathLogged.exchange(true, std::memory_order_relaxed))
        {
            LOG_INFO("FusedNvfp4GemmAllReducePlugin: using GEMM + ncclAllReduce path");
        }

        cudaError_t err = mGemmRunner->run(inputs[kFP4_ACT_IDX], inputs[kFP4_WEIGHT_IDX], globalActScaleTiled,
            weightSFB, outputs[kOUT_TENSOR_IDX], m, N, K, stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("FusedNvfp4GemmAllReducePlugin: FP16 CuTe DSL GEMM (NCCL fallback) failed: %s",
                cudaGetErrorString(err));
            return -1;
        }

        void* ncclComm = nullptr;
        void* ncclFnRaw = nullptr;
        getNcclRegistrationForDevice(currentDevice, &ncclComm, &ncclFnRaw);
        auto ncclFn = reinterpret_cast<NcclAllReduceFn>(ncclFnRaw);
        if (ncclComm && ncclFn && mTpSize > 1)
        {
            int32_t ncclResult = ncclFn(outputs[kOUT_TENSOR_IDX], outputs[kOUT_TENSOR_IDX], outElements, kNcclFloat16,
                kNcclSum, ncclComm, stream);
            if (ncclResult != kNcclSuccess)
            {
                LOG_ERROR("FusedNvfp4GemmAllReducePlugin: NCCL allReduce failed (%d)", ncclResult);
                return -1;
            }
        }
        return 0;
#endif // SUPPORTS_FP8
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("FusedNvfp4GemmAllReducePlugin enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t FusedNvfp4GemmAllReducePlugin::onShapeChange(
    PluginTensorDesc const* /* in */, int32_t nbInputs, PluginTensorDesc const* /* out */, int32_t nbOutputs) noexcept
{
    return (nbInputs == 5 && nbOutputs == 1) ? 0 : -1;
}

IPluginV3* FusedNvfp4GemmAllReducePlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* FusedNvfp4GemmAllReducePlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("tp_size", &mTpSize, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

char const* FusedNvfp4GemmAllReducePlugin::getPluginName() const noexcept
{
    return kFUSED_NVFP4_GEMM_AR_PLUGIN_NAME;
}

char const* FusedNvfp4GemmAllReducePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void FusedNvfp4GemmAllReducePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* FusedNvfp4GemmAllReducePlugin::getPluginVersion() const noexcept
{
    return kFUSED_NVFP4_GEMM_AR_PLUGIN_VERSION;
}

bool FusedNvfp4GemmAllReducePlugin::ensureGemmRunners() noexcept
{
    try
    {
        if (!kernels::CuteDslGemmNvFp4Runner::loadKernelModules())
        {
            LOG_ERROR("FusedNvfp4GemmAllReducePlugin: failed to load CuTe DSL NVFP4 GEMM modules");
            return false;
        }
        if (mGemmRunner == nullptr)
        {
            mGemmRunner = new kernels::CuteDslGemmNvFp4Runner(/*mmaTilerN=*/64);
        }
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("FusedNvfp4GemmAllReducePlugin initialize failed: %s", e.what());
        return false;
    }
}

// ========================== Creator Implementation ==========================

FusedNvfp4GemmAllReducePluginCreator::FusedNvfp4GemmAllReducePluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("tp_size", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* FusedNvfp4GemmAllReducePluginCreator::getPluginName() const noexcept
{
    return kFUSED_NVFP4_GEMM_AR_PLUGIN_NAME;
}

PluginFieldCollection const* FusedNvfp4GemmAllReducePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void FusedNvfp4GemmAllReducePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* FusedNvfp4GemmAllReducePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* FusedNvfp4GemmAllReducePluginCreator::getPluginVersion() const noexcept
{
    return kFUSED_NVFP4_GEMM_AR_PLUGIN_VERSION;
}

IPluginV3* FusedNvfp4GemmAllReducePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        auto* plugin = new FusedNvfp4GemmAllReducePlugin(name, fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("FusedNvfp4GemmAllReducePluginCreator::createPlugin failed: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
