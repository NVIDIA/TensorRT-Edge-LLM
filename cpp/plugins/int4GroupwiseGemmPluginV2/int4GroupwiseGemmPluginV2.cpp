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

#include "int4GroupwiseGemmPluginV2.h"
#include "common/cudaUtils.h"
#include "cuteDslInt4Gemm.h"
#include "cuteDslInt4Gemv.h"

#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>

using namespace nvinfer1;
namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kINT4_GEMM_V2_PLUGIN_VERSION{"1"};
constexpr char const* kINT4_GEMM_V2_PLUGIN_NAME{"Int4GroupwiseGemmPluginV2"};
constexpr char const* kMAX_LOCK_WORKSPACE_BYTES_FIELD{"max_lock_workspace_bytes"};
// Worst-case lock sizing follows the current CuTe DSL kernel set: bM=16 is
// the smallest M tile and bN=128 is fixed across variants.
constexpr int32_t kWORST_CASE_LOCK_TILE_M{16};
constexpr int32_t kLOCK_TILE_N{128};

// Fragment weight buffer rows for a (N, K) problem (bN=128, bK=64, kn=8).
// The plugin's INT8 input[1] has shape [rows, 512] (each row = 128 uint32 words).
inline int32_t fragmentRows(int32_t N, int32_t K)
{
    return divUp(N, 128) * divUp(K, 64) * 8;
}

#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED
inline int64_t computeMaxLockWorkspaceBytes(
    DynamicPluginTensorDesc const* inputs, int32_t nbInputs, int32_t N, int32_t K)
{
    if (inputs == nullptr || nbInputs < 1 || N <= 0 || K <= 0 || K % 64 != 0)
    {
        return 0;
    }
    auto const& mx = inputs[0].max;
    if (mx.nbDims < 2)
    {
        return 0;
    }
    int64_t const mMax = static_cast<int64_t>(mx.d[0]) * mx.d[1];
    if (mMax <= 0)
    {
        return 0;
    }
    int64_t const locks = cuteDslInt4LockCount(static_cast<int32_t>(mMax), N, kWORST_CASE_LOCK_TILE_M, kLOCK_TILE_N);
    return locks * static_cast<int64_t>(sizeof(int32_t));
}
#endif
} // namespace

// Static class fields initialization
PluginFieldCollection Int4GroupwiseGemmPluginV2Creator::mFieldCollection{};
std::vector<PluginField> Int4GroupwiseGemmPluginV2Creator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Int4GroupwiseGemmPluginV2Creator);

Int4GroupwiseGemmPluginV2::Int4GroupwiseGemmPluginV2(std::string const& name, int32_t N, int32_t K, int32_t groupSize)
    : mLayerName(name)
    , mGemmN(N)
    , mGemmK(K)
    , mGroupSize(groupSize)
{
}

Int4GroupwiseGemmPluginV2::Int4GroupwiseGemmPluginV2(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName == "gemm_n")
        {
            mGemmN = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "gemm_k")
        {
            mGemmK = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "group_size")
        {
            mGroupSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == kMAX_LOCK_WORKSPACE_BYTES_FIELD)
        {
            mMaxLockWorkspaceBytes = *static_cast<int64_t const*>(fc->fields[i].data);
        }
    }
}

Int4GroupwiseGemmPluginV2::~Int4GroupwiseGemmPluginV2()
{
    releaseLockWorkspace();
}

IPluginCapability* Int4GroupwiseGemmPluginV2::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Int4GroupwiseGemmPluginV2::clone() noexcept
{
    try
    {
        auto* plugin = new Int4GroupwiseGemmPluginV2(mLayerName, mGemmN, mGemmK, mGroupSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->mTactic = mTactic;
        plugin->mAutotuneM = mAutotuneM;
        plugin->mMaxLockWorkspaceBytes = mMaxLockWorkspaceBytes;
        return plugin;
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

char const* Int4GroupwiseGemmPluginV2::getPluginName() const noexcept
{
    return kINT4_GEMM_V2_PLUGIN_NAME;
}

char const* Int4GroupwiseGemmPluginV2::getPluginVersion() const noexcept
{
    return kINT4_GEMM_V2_PLUGIN_VERSION;
}

char const* Int4GroupwiseGemmPluginV2::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Int4GroupwiseGemmPluginV2::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

int32_t Int4GroupwiseGemmPluginV2::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Int4GroupwiseGemmPluginV2::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    DataType const* /* inputTypes */, int32_t /* nbInputs */) const noexcept
{
    try
    {
        assert(nbOutputs == 1);
        outputTypes[0] = DataType::kHALF;
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t Int4GroupwiseGemmPluginV2::getOutputShapes(DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs,
    [[maybe_unused]] int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        assert(nbInputs == 3);
        assert(nbOutputs == 1);
        outputs[0].nbDims = 3;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = inputs[0].d[1];
        outputs[0].d[2] = exprBuilder.constant(mGemmN);
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool Int4GroupwiseGemmPluginV2::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    try
    {
        assert(nbInputs == 3 && nbOutputs == 1);
        assert(pos < (nbInputs + nbOutputs));
        auto const& tensorDesc = inOut[pos].desc;
        bool status{true};

        switch (pos)
        {
        case 0:
        {
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == PluginFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 3;
            status &= tensorDesc.dims.d[2] == mGemmK;
            break;
        }
        case 1:
        {
            // Fragment-layout weights: INT8 [rows, 512], rows = ceil(N/128)*ceil(K/64)*8.
            status &= tensorDesc.type == DataType::kINT8;
            status &= tensorDesc.format == PluginFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 2;
            status &= tensorDesc.dims.d[0] == fragmentRows(mGemmN, mGemmK);
            status &= tensorDesc.dims.d[1] == 512;
            break;
        }
        case 2:
        {
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == PluginFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 2;
            status &= tensorDesc.dims.d[0] == mGemmK / mGroupSize;
            status &= tensorDesc.dims.d[1] == mGemmN;
            break;
        }
        case 3:
        {
            status &= tensorDesc.type == DataType::kHALF;
            status &= tensorDesc.format == PluginFormat::kLINEAR;
            status &= tensorDesc.dims.nbDims == 3;
            status &= tensorDesc.dims.d[2] == mGemmN;
            break;
        }
        default: break;
        }
        return status;
    }
    catch (std::exception const& e)
    {
        return false;
    }
}

int32_t Int4GroupwiseGemmPluginV2::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    // Capture the profile's optimum token count M (= batch*seq of activation input[0]).
    // TensorRT autotunes at this representative shape, so it is the right M for the
    // CTA-tile pruning heuristic in getValidTactics/getNbTactics.
    if (nbInputs >= 1)
    {
        auto const& opt = in[0].opt;
        if (opt.nbDims >= 2)
        {
            int64_t const m = static_cast<int64_t>(opt.d[0]) * opt.d[1];
            mAutotuneM = (m > 0) ? static_cast<int32_t>(m) : 0;
        }
    }
#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED
    int64_t const maxLockWorkspaceBytes = computeMaxLockWorkspaceBytes(in, nbInputs, mGemmN, mGemmK);
    if (maxLockWorkspaceBytes > mMaxLockWorkspaceBytes)
    {
        mMaxLockWorkspaceBytes = maxLockWorkspaceBytes;
    }
#endif
    return 0;
}

size_t Int4GroupwiseGemmPluginV2::getWorkspaceSize([[maybe_unused]] DynamicPluginTensorDesc const* inputs,
    int32_t /* nbInputs */, DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

bool Int4GroupwiseGemmPluginV2::ensureLockWorkspace(size_t requiredBytes) noexcept
{
#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED
    if (requiredBytes <= mLockWorkspaceBytes)
    {
        return true;
    }

    void* newWorkspace{nullptr};
    cudaError_t const allocError = cudaMalloc(&newWorkspace, requiredBytes);
    if (allocError != cudaSuccess)
    {
        LOG_ERROR("Int4GroupwiseGemmPluginV2: cudaMalloc for serial split-K locks failed: %s",
            cudaGetErrorString(allocError));
        return false;
    }

    releaseLockWorkspace();
    mLockWorkspace = newWorkspace;
    mLockWorkspaceBytes = requiredBytes;
    return true;
#else
    return true;
#endif
}

void Int4GroupwiseGemmPluginV2::releaseLockWorkspace() noexcept
{
    if (mLockWorkspace == nullptr)
    {
        return;
    }
    cudaError_t const freeError = cudaFree(mLockWorkspace);
    if (freeError != cudaSuccess)
    {
        LOG_ERROR(
            "Int4GroupwiseGemmPluginV2: cudaFree for serial split-K locks failed: %s", cudaGetErrorString(freeError));
    }
    mLockWorkspace = nullptr;
    mLockWorkspaceBytes = 0;
}

int32_t Int4GroupwiseGemmPluginV2::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, [[maybe_unused]] void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        auto const& inputDesc0 = inputDesc[0];
        [[maybe_unused]] int32_t const M = inputDesc0.dims.d[0] * inputDesc0.dims.d[1];

#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED
        // Fragment-layout contract for BOTH the GEMV and GEMM paths: input[1]
        // stores ceil(N/128) N tiles and K/64 K tiles. Both kernels predicate
        // the final N tile and require a complete K tile.
        if (mGemmN <= 0 || mGemmK <= 0 || mGemmK % 64 != 0)
        {
            return -1;
        }

        // Decode regime: route small M to the CUDA-core GEMV. At small M the GEMM's
        // tensor-core MMA underfills its M rows, so the GEMV is the better compute
        // path. It reads the SAME fragment weight buffer (input[1]), so there is no
        // extra weight and no repack -- one buffer serves prefill and decode.
        // Threshold kGemvDispatchMaxM=4 (the tensor-core crossover); the kernel
        // itself supports up to kGemvMaxM=8.
        constexpr int32_t kGemvDispatchMaxM = 4;
        if (M >= 1 && M <= kGemvDispatchMaxM && cuteDslInt4GemvSupported(M) && cuteDslInt4GemvLoadModules())
        {
            return cuteDslInt4GemvLaunch(M, inputs[0], inputs[1], inputs[2], outputs[0], mGemmN, mGemmK, stream);
        }
#endif

#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED
        if (!cuteDslInt4GemmLoadModules())
        {
            return -1;
        }
        // Tactic id = variant index + 1; tactic 0 (default) maps to variant 0,
        // which is always valid for an eligible positive-N, K%64==0 problem.
        int32_t const idx = (mTactic > 0) ? (mTactic - 1) : 0;
        if (idx >= cuteDslInt4NumVariants())
        {
            return -1;
        }
        CuteDslInt4Variant const v = cuteDslInt4VariantAt(idx);
        if (v.splitK > 1)
        {
            int64_t const nLocks = cuteDslInt4LockCount(M, mGemmN, v.bM, v.bN);
            size_t const lockBytes = static_cast<size_t>(nLocks) * sizeof(int32_t);
            if (mLockWorkspace == nullptr || lockBytes > mLockWorkspaceBytes)
            {
                LOG_ERROR("Int4GroupwiseGemmPluginV2: serial split-K lock buffer was not allocated for this shape.");
                return -1;
            }
            cudaError_t const memsetError = cudaMemsetAsync(mLockWorkspace, 0, lockBytes, stream);
            if (memsetError != cudaSuccess)
            {
                LOG_ERROR("Int4GroupwiseGemmPluginV2: cudaMemsetAsync for serial split-K locks failed: %s",
                    cudaGetErrorString(memsetError));
                return -1;
            }
        }
        // input[1] is the fragment-layout weight buffer (INT8 bytes of the uint32
        // words); pass it straight through as mQW -- no repack, no cache.
        return cuteDslInt4GemmLaunch(v, inputs[0], inputs[1], inputs[2], outputs[0],
            (v.splitK > 1) ? mLockWorkspace : nullptr, M, mGemmN, mGemmK, /*swizzle=*/1, stream);
#else
        return -1;
#endif
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t Int4GroupwiseGemmPluginV2::onShapeChange(PluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    PluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED
    try
    {
        if (mMaxLockWorkspaceBytes <= 0)
        {
            return -1;
        }
        return ensureLockWorkspace(static_cast<size_t>(mMaxLockWorkspaceBytes)) ? 0 : -1;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
#endif
    return 0;
}

IPluginV3* Int4GroupwiseGemmPluginV2::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED
namespace
{
// Combined per-shape tactic filter shared by getNbTactics()/getValidTactics() so the
// two always agree on the same set. A variant is autotuned iff:
//  (1) it can serve this (N, K): N>0, K%64==0, splitK | ceil(K/bK)  [validity], AND
//  (2) its CTA tile passes the 16-variant subset per-M pruning
//      derived by set-cover analysis.
inline bool tacticSelectable(CuteDslInt4Variant const& v, int32_t N, int32_t K, int32_t autotuneM)
{
    if (!cuteDslInt4VariantValid(v, N, K))
    {
        return false;
    }
    if (autotuneM <= 0)
    {
        return true;
    }
    return (autotuneM >= 256) ? (v.bM != 16) : (v.bM != 128);
}
} // namespace
#endif

int32_t Int4GroupwiseGemmPluginV2::getNbTactics() noexcept
{
#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED
    int32_t count = 0;
    for (int32_t i = 0; i < cuteDslInt4NumVariants(); ++i)
    {
        if (tacticSelectable(cuteDslInt4VariantAt(i), mGemmN, mGemmK, mAutotuneM))
        {
            ++count;
        }
    }
    return count;
#else
    return 0;
#endif
}

int32_t Int4GroupwiseGemmPluginV2::getValidTactics(
    [[maybe_unused]] int32_t* tactics, [[maybe_unused]] int32_t nbTactics) noexcept
{
#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED
    // Tactic id = variant table index + 1 (positive; 0 is TRT's reserved default).
    int32_t j = 0;
    for (int32_t i = 0; i < cuteDslInt4NumVariants() && j < nbTactics; ++i)
    {
        if (tacticSelectable(cuteDslInt4VariantAt(i), mGemmN, mGemmK, mAutotuneM))
        {
            tactics[j++] = i + 1;
        }
    }
#endif
    return 0;
}

int32_t Int4GroupwiseGemmPluginV2::setTactic(int32_t tactic) noexcept
{
#ifdef CUTE_DSL_INT4_FP16_GEMM_ENABLED
    if (tactic > 0)
    {
        int32_t const idx = tactic - 1;
        if (idx >= cuteDslInt4NumVariants())
        {
            return -1;
        }
        CuteDslInt4Variant const v = cuteDslInt4VariantAt(idx);
        if (!cuteDslInt4VariantValid(v, mGemmN, mGemmK))
        {
            return -1;
        }
    }
#endif
    mTactic = tactic;
    return 0;
}

char const* Int4GroupwiseGemmPluginV2::getTimingCacheID() noexcept
{
    // Opt this plugin into TRT's build timing cache (non-null enables it). An empty
    // suffix suffices: the plugin's whole state (N, K, group_size) is already carried
    // by the I/O tensor shapes, which TRT auto-hashes as the timing-cache key prefix.
    // Effect is arch-dependent: where TRT keeps the int4 ops as standalone layers
    // (currently SM<100), identical-shape layers share a timing-cache entry and are
    // autotuned once. Where the ops are fused into a Myelin ForeignNode (currently SM>=100),
    // Myelin's own autotuner does not consult this cache, so it is simply inert.
    return "";
}

PluginFieldCollection const* Int4GroupwiseGemmPluginV2::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("gemm_n", &mGemmN, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("gemm_k", &mGemmK, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("group_size", &mGroupSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(kMAX_LOCK_WORKSPACE_BYTES_FIELD, &mMaxLockWorkspaceBytes, PluginFieldType::kINT64, 1);

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

Int4GroupwiseGemmPluginV2Creator::Int4GroupwiseGemmPluginV2Creator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("gemm_n", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("gemm_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group_size", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Int4GroupwiseGemmPluginV2Creator::getPluginName() const noexcept
{
    return kINT4_GEMM_V2_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* Int4GroupwiseGemmPluginV2Creator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void Int4GroupwiseGemmPluginV2Creator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* Int4GroupwiseGemmPluginV2Creator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* Int4GroupwiseGemmPluginV2Creator::getPluginVersion() const noexcept
{
    return kINT4_GEMM_V2_PLUGIN_VERSION;
}

IPluginV3* Int4GroupwiseGemmPluginV2Creator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        Int4GroupwiseGemmPluginV2* plugin = new Int4GroupwiseGemmPluginV2(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
