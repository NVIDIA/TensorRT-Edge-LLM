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

#include "decoderXQARunner.h"
#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "xqaKernelTypes.h"

#include <algorithm>
#include <array>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;

using XQADataType = xqa::kernels::Data_type;
using XQAKernelMetaInfo = xqa::kernels::XQAKernelMetaInfo;
using XQAKernelVariant = XQAKernelMetaInfo::XQAKernelVariant;

namespace
{

constexpr uint32_t kHEAD_DIM_512{512U};
constexpr uint32_t kSPLIT_HEAD_DIM_512_CLUSTER_SIZE{2U};
constexpr uint32_t kSPLIT_HEAD_DIM_512_CTA_DIM_Z{2U};
constexpr uint32_t kDEFAULT_XQA_CTA_DIM_Z{2U};
constexpr uint32_t kDEFAULT_XQA_CTA_DIM_X{128U};
constexpr uint32_t kHEAD_DIM_512_CTA_DIM_X{256U};
// Max vanilla XQA kernel argument count. No-sliding kernels use one fewer argument.
constexpr size_t kXQA_KERNEL_PARAM_COUNT{12U};
// Max spec-decode XQA kernel argument count. No-sliding kernels use one fewer argument.
constexpr size_t kSPEC_DECODE_XQA_KERNEL_PARAM_COUNT{16U};

//! @throws std::runtime_error if datatype is unsupported
XQADataType trtToXqaDataType(nvinfer1::DataType type)
{
    XQADataType xqaType{XQADataType::DATA_TYPE_FP16};
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: xqaType = XQADataType::DATA_TYPE_FP32; break;
    case nvinfer1::DataType::kHALF: xqaType = XQADataType::DATA_TYPE_FP16; break;
    case nvinfer1::DataType::kBF16: xqaType = XQADataType::DATA_TYPE_BF16; break;
    case nvinfer1::DataType::kFP8: xqaType = XQADataType::DATA_TYPE_E4M3; break;
    default: throw std::runtime_error("Unsupported datatype for XQA.");
    }
    return xqaType;
}
struct XQAKernelLoadHashKey
{
    XQADataType data_type;
    XQADataType kv_data_type;
    int32_t sm;
    bool specDecode;
    bool pagedKVCache;
    int32_t deviceId; //!< CUDA device ID for per-device kernel isolation.

    bool operator==(XQAKernelLoadHashKey const& other) const noexcept
    {
        return data_type == other.data_type && kv_data_type == other.kv_data_type && sm == other.sm
            && specDecode == other.specDecode && pagedKVCache == other.pagedKVCache && deviceId == other.deviceId;
    }
};

struct XQAKernelLoadHasher
{
    size_t operator()(XQAKernelLoadHashKey const& s) const noexcept
    {
        size_t key = s.data_type;
        key <<= 16;
        key ^= s.kv_data_type;
        key <<= 16;
        key ^= s.sm;
        key <<= 4;
        key ^= s.specDecode;
        key <<= 4;
        key ^= s.pagedKVCache;
        key ^= (static_cast<size_t>(s.deviceId) << 24);
        return key;
    }
};

struct XQAKernelRuntimeHashKey
{
    XQADataType q_data_type;
    XQADataType kv_data_type;
    int32_t head_size;
    int32_t num_q_heads_per_kv;
    int32_t beam_size;
    bool sliding_window;
    int32_t tokens_per_page;

    bool operator==(XQAKernelRuntimeHashKey const& other) const noexcept
    {
        return q_data_type == other.q_data_type && kv_data_type == other.kv_data_type && head_size == other.head_size
            && num_q_heads_per_kv == other.num_q_heads_per_kv && beam_size == other.beam_size
            && sliding_window == other.sliding_window && tokens_per_page == other.tokens_per_page;
    }
};

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParams(XQALaunchParams const& xqaParams) noexcept
{
    constexpr int32_t kBEAM_SIZE{1};
    int32_t numQHeadPerKV = xqaParams.numQheads / xqaParams.numKVheads;
    return {trtToXqaDataType(xqaParams.dataType), trtToXqaDataType(xqaParams.kvDataType), xqaParams.headSize,
        numQHeadPerKV, kBEAM_SIZE, xqaParams.slidingWinSize > 0, static_cast<int32_t>(xqaParams.kvCache.tokensPerPage)};
}

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParamsSpecDecode(XQALaunchParams const& xqaParams) noexcept
{
    constexpr int32_t kBEAM_SIZE{1};
    constexpr int32_t kQHEAD_PER_KV = 0; // Tree attention kernel supports any ratio of Q/KV heads.
    return {trtToXqaDataType(xqaParams.dataType), trtToXqaDataType(xqaParams.kvDataType), xqaParams.headSize,
        kQHEAD_PER_KV, kBEAM_SIZE, xqaParams.slidingWinSize > 0, static_cast<int32_t>(xqaParams.kvCache.tokensPerPage)};
}

std::string formatMissingXQAKernelMessage(char const* kernelName, XQAKernelRuntimeHashKey const& hashKey,
    XQALaunchParams const& params, int32_t smVersion, bool specDecode)
{
    return format::fmtstr(
        "No available cubin for %s. Runtime key: sm=%d, q_dtype=%d, kv_dtype=%d, head_size=%d, "
        "q_heads_per_kv=%d, beam_size=%d, sliding_window=%d. Launch params: q_heads=%d, kv_heads=%d, batch_size=%d, "
        "kv_cache_capacity=%u, q_seq_len=%d, head_group_size=%d, trt_dtype=%d, trt_kv_dtype=%d. "
        "Expected JIT cubin key: sm=%d, data_type=%d, kv_data_type=%d, head_size=%d, q_heads_per_kv=%d, "
        "sliding_window=%d, spec_decode=%d.",
        kernelName, smVersion, static_cast<int32_t>(hashKey.q_data_type), static_cast<int32_t>(hashKey.kv_data_type),
        hashKey.head_size, hashKey.num_q_heads_per_kv, hashKey.beam_size, static_cast<int32_t>(hashKey.sliding_window),
        params.numQheads, params.numKVheads, params.batchSize, params.kvCache.capacity, params.qSeqLen,
        params.headGroupSize, static_cast<int32_t>(params.dataType), static_cast<int32_t>(params.kvDataType), smVersion,
        static_cast<int32_t>(params.dataType), static_cast<int32_t>(params.kvDataType), hashKey.head_size,
        hashKey.num_q_heads_per_kv, static_cast<int32_t>(hashKey.sliding_window), static_cast<int32_t>(specDecode));
}

struct XQAKernelRuntimeHasher
{
    size_t operator()(XQAKernelRuntimeHashKey const& s) const noexcept
    {
        size_t key = s.q_data_type;
        key <<= 16;
        key ^= s.kv_data_type;
        key <<= 16;
        key ^= s.head_size;
        key <<= 8;
        key ^= s.num_q_heads_per_kv;
        key <<= 8;
        key ^= s.beam_size;
        key <<= 4;
        key ^= s.sliding_window;
        key <<= 8;
        key ^= s.tokens_per_page;
        return key;
    }
};

struct XQAKernelFuncInfo
{
    uint32_t mSharedMemBytes{0};
    CUfunction mDeviceFunction{0};
    uint32_t mHeadDim{0};
    uint32_t mMTileSize{0};
    uint32_t mSMVersion{0};
    XQAKernelVariant mKernelVariant{XQAKernelMetaInfo::KERNEL_VARIANT_STANDARD};
    bool mRequiresClusterLaunch{false};
    bool mRequiresDistributedSharedMemory{false};
    bool mSlidingWindow{false};
};

struct XQADeviceCapability
{
    uint32_t mMaxSharedMemPerBlockOptin{0};
    bool mSupportsClusterLaunch{false};
    bool mSupportsDistributedSharedMemory{false};
};

struct KernelContiguousKVCache
{
    void* data{nullptr};
    int32_t const* sequenceLengths{nullptr};
    uint32_t capacity{0};
};

struct KernelPagedKVCache
{
    void* data{nullptr};
    int32_t const* pageList{nullptr};
    int32_t const* sequenceLengths{nullptr};
    uint32_t maxNbPagesPerSeq{0};
};

using KernelKVCacheArg = std::variant<KernelContiguousKVCache, KernelPagedKVCache>;

bool usePagedKVCache(XQALaunchParams::KVCache const& kvCache) noexcept
{
    return kvCache.tokensPerPage != 0;
}

bool isPowerOfTwo(uint32_t value) noexcept
{
    return value > 0U && (value & (value - 1U)) == 0U;
}

uint32_t getMaxNbPagesPerSeq(XQALaunchParams::KVCache const& kvCache)
{
    check::check(kvCache.tokensPerPage > 0, "Paged KV cache requires non-zero tokensPerPage.");
    check::check(isPowerOfTwo(kvCache.tokensPerPage), "Paged KV cache tokensPerPage must be a power of 2.");
    check::check(
        kvCache.capacity % kvCache.tokensPerPage == 0U, "Paged KV cache capacity must be divisible by tokensPerPage.");
    return kvCache.capacity / kvCache.tokensPerPage;
}

KernelKVCacheArg makeKernelKVCacheArg(XQALaunchParams::KVCache const& kvCache, bool usePaged)
{
    if (!usePaged)
    {
        return KernelContiguousKVCache{kvCache.data, kvCache.sequence_lengths, kvCache.capacity};
    }

    return KernelPagedKVCache{kvCache.data, kvCache.pageList, kvCache.sequence_lengths, getMaxNbPagesPerSeq(kvCache)};
}

void* getKernelKVCacheArg(KernelKVCacheArg& kvCacheArg)
{
    return std::visit([](auto& kernelKVCache) -> void* { return &kernelKVCache; }, kvCacheArg);
}

XQADeviceCapability getDeviceCapability()
{
    constexpr int32_t kDEVICE_ID{0};
    CUdevice cuDevice{};
    CUDA_DRIVER_CHECK(cuDeviceGet(&cuDevice, kDEVICE_ID));

    int32_t maxSharedMemPerBlockOptin{0};
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(
        &maxSharedMemPerBlockOptin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cuDevice));
    check::check(maxSharedMemPerBlockOptin > 0, "Failed to get max shared memory per block opt-in.");

    bool supportsClusterLaunch{false};
#if SUPPORTS_CLUSTER_LAUNCH
    int32_t clusterLaunch{0};
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&clusterLaunch, CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH, cuDevice));
    supportsClusterLaunch = clusterLaunch != 0;
#endif // SUPPORTS_CLUSTER_LAUNCH

    return {static_cast<uint32_t>(maxSharedMemPerBlockOptin), supportsClusterLaunch, supportsClusterLaunch};
}

bool isKernelCompatibleWithDevice(
    XQAKernelFuncInfo const& kernelInfo, XQADeviceCapability const& deviceCapability) noexcept
{
    if (kernelInfo.mSharedMemBytes > deviceCapability.mMaxSharedMemPerBlockOptin)
    {
        return false;
    }
    if (kernelInfo.mRequiresClusterLaunch && !deviceCapability.mSupportsClusterLaunch)
    {
        return false;
    }
    if (kernelInfo.mRequiresDistributedSharedMemory && !deviceCapability.mSupportsDistributedSharedMemory)
    {
        return false;
    }
    return true;
}

uint32_t getKernelVariantPriority(XQAKernelVariant variant) noexcept
{
    // Lower priority value is preferred when multiple compatible candidates exist for the same runtime key.
    switch (variant)
    {
    case XQAKernelMetaInfo::KERNEL_VARIANT_STANDARD: return 0U;
    case XQAKernelMetaInfo::KERNEL_VARIANT_FULL_SMEM_HEAD_DIM512: return 0U;
    case XQAKernelMetaInfo::KERNEL_VARIANT_FULL_SMEM_HEAD_DIM512_ROW_MAX_METHOD4: return 1U;
    case XQAKernelMetaInfo::KERNEL_VARIANT_2CTA_HEAD_DIM512: return 2U;
    case XQAKernelMetaInfo::KERNEL_VARIANT_TILED_QKV_STAGING_HEAD_DIM512: return 3U;
    }
    return 3U;
}

uint32_t getCtaDimX(XQAKernelFuncInfo const& kernelInfo) noexcept
{
    return kernelInfo.mHeadDim == kHEAD_DIM_512 && !kernelInfo.mRequiresClusterLaunch ? kHEAD_DIM_512_CTA_DIM_X
                                                                                      : kDEFAULT_XQA_CTA_DIM_X;
}

XQAKernelVariant getJitKernelVariant(XQAJitKey const& key) noexcept
{
    if (key.headSize != static_cast<int32_t>(kHEAD_DIM_512))
    {
        return XQAKernelMetaInfo::KERNEL_VARIANT_STANDARD;
    }
    if (key.sm == 80 || key.sm == 87)
    {
        return XQAKernelMetaInfo::KERNEL_VARIANT_FULL_SMEM_HEAD_DIM512_ROW_MAX_METHOD4;
    }
    if (key.sm == 86 || key.sm == 89)
    {
        return XQAKernelMetaInfo::KERNEL_VARIANT_TILED_QKV_STAGING_HEAD_DIM512;
    }
    if (key.sm == 120 || key.sm == 121)
    {
        return XQAKernelMetaInfo::KERNEL_VARIANT_2CTA_HEAD_DIM512;
    }
    return XQAKernelMetaInfo::KERNEL_VARIANT_FULL_SMEM_HEAD_DIM512;
}

bool jitRequiresClusterLaunch(XQAJitKey const& key) noexcept
{
    return getJitKernelVariant(key) == XQAKernelMetaInfo::KERNEL_VARIANT_2CTA_HEAD_DIM512;
}

bool isJitKernelSupportedByBuild(XQAJitKey const& key) noexcept
{
#if SUPPORTS_CLUSTER_LAUNCH
    (void) key;
    return true;
#else
    return !jitRequiresClusterLaunch(key);
#endif // SUPPORTS_CLUSTER_LAUNCH
}

#if SUPPORTS_CLUSTER_LAUNCH
void launch2CtaHeadDim512ClusterKernel(XQAKernelFuncInfo const& kernelInfo, dim3 const& dimGrid, dim3 const& dimCta,
    cudaStream_t const& stream, void** kernelParams)
{
    CUlaunchAttribute launchAttr{};
    launchAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    launchAttr.value.clusterDim.x = kSPLIT_HEAD_DIM_512_CLUSTER_SIZE;
    launchAttr.value.clusterDim.y = 1U;
    launchAttr.value.clusterDim.z = 1U;

    CUlaunchConfig launchConfig{};
    launchConfig.gridDimX = dimGrid.x;
    launchConfig.gridDimY = dimGrid.y;
    launchConfig.gridDimZ = dimGrid.z;
    launchConfig.blockDimX = dimCta.x;
    launchConfig.blockDimY = dimCta.y;
    launchConfig.blockDimZ = dimCta.z;
    launchConfig.sharedMemBytes = kernelInfo.mSharedMemBytes;
    launchConfig.hStream = stream;
    launchConfig.attrs = &launchAttr;
    launchConfig.numAttrs = 1U;

    CUDA_DRIVER_CHECK(cuLaunchKernelEx(&launchConfig, kernelInfo.mDeviceFunction, kernelParams, nullptr));
}
#endif // SUPPORTS_CLUSTER_LAUNCH

uint32_t getXQAKernelGridDimX(XQAKernelFuncInfo const& kernelInfo) noexcept
{
    return kernelInfo.mRequiresClusterLaunch ? kSPLIT_HEAD_DIM_512_CLUSTER_SIZE : 1U;
}

dim3 getXQAKernelCtaDim(XQAKernelFuncInfo const& kernelInfo) noexcept
{
    return dim3{getCtaDimX(kernelInfo), 1U,
        kernelInfo.mRequiresClusterLaunch ? kSPLIT_HEAD_DIM_512_CTA_DIM_Z : kDEFAULT_XQA_CTA_DIM_Z};
}

void launchXQAKernel(XQAKernelFuncInfo const& kernelInfo, dim3 const& dimGrid, dim3 const& dimCta,
    cudaStream_t const& stream, void** kernelParams)
{
    bool const useClusterLaunch = kernelInfo.mRequiresClusterLaunch;
#if SUPPORTS_CLUSTER_LAUNCH
    // Keep the cluster and regular launch paths shared by decode and spec-decode dispatch.
    if (useClusterLaunch)
    {
        launch2CtaHeadDim512ClusterKernel(kernelInfo, dimGrid, dimCta, stream, kernelParams);
        return;
    }
#else
    check::check(!useClusterLaunch, "XQA head_dim=512 2CTA cluster kernel is unavailable.");
#endif // SUPPORTS_CLUSTER_LAUNCH
    CUDA_DRIVER_CHECK(cuLaunchKernel(kernelInfo.mDeviceFunction, dimGrid.x, dimGrid.y, dimGrid.z, dimCta.x, dimCta.y,
        dimCta.z, kernelInfo.mSharedMemBytes, stream, kernelParams, nullptr));
}

struct CudaModuleDeleter
{
    void operator()(std::remove_pointer_t<CUmodule>* module) const noexcept
    {
        if (module != nullptr)
        {
            (void) cuModuleUnload(module);
        }
    }
};

using CudaModulePtr = std::unique_ptr<std::remove_pointer_t<CUmodule>, CudaModuleDeleter>;

using XQAKernelParams = std::array<void*, kXQA_KERNEL_PARAM_COUNT>;
using SpecDecodeXQAKernelParams = std::array<void*, kSPEC_DECODE_XQA_KERNEL_PARAM_COUNT>;

XQAKernelParams makeXQAKernelParams(XQALaunchParams& params, bool const slidingWindow, void* const kvCacheArg)
{
    XQAKernelParams kernelParams{};
    size_t idx{0U};
    kernelParams[idx++] = &params.numKVheads;
    if (slidingWindow)
    {
        kernelParams[idx++] = &params.slidingWinSize;
    }
    kernelParams[idx++] = &params.attentionScale;
    kernelParams[idx++] = &params.output;
    kernelParams[idx++] = &params.qInputPtr;
    kernelParams[idx++] = &params.attentionSinks;
    kernelParams[idx++] = kvCacheArg;
    kernelParams[idx++] = &params.batchSize;
    kernelParams[idx++] = &params.kScale;
    kernelParams[idx++] = &params.vScale;
    kernelParams[idx++] = &params.semaphores;
    kernelParams[idx++] = &params.scratch;
    return kernelParams;
}

SpecDecodeXQAKernelParams makeSpecDecodeXQAKernelParams(
    XQALaunchParams& params, bool const slidingWindow, void* const kvCacheArg)
{
    SpecDecodeXQAKernelParams kernelParams{};
    size_t idx{0U};
    kernelParams[idx++] = &params.qSeqLen;
    kernelParams[idx++] = &params.numKVheads;
    kernelParams[idx++] = &params.headGroupSize;
    kernelParams[idx++] = &params.qCuSeqLen;
    if (slidingWindow)
    {
        kernelParams[idx++] = &params.slidingWinSize;
    }
    kernelParams[idx++] = &params.attentionScale;
    kernelParams[idx++] = &params.output;
    kernelParams[idx++] = &params.qInputPtr;
    kernelParams[idx++] = &params.treeAttnMask;
    kernelParams[idx++] = &params.attentionSinks;
    kernelParams[idx++] = kvCacheArg;
    kernelParams[idx++] = &params.batchSize;
    kernelParams[idx++] = &params.kScale;
    kernelParams[idx++] = &params.vScale;
    kernelParams[idx++] = &params.semaphores;
    kernelParams[idx++] = &params.scratch;
    return kernelParams;
}

class XQAKernelList
{
public:
    XQAKernelList() noexcept = default;
    ~XQAKernelList() noexcept = default;

    //! @throws std::runtime_error if a CUDA driver error occurs
    bool loadJitKernel(XQAJitKey const& key, void const* cubinData, size_t cubinSize)
    {
        check::check(cubinData != nullptr && cubinSize > 0, "Invalid XQA JIT cubin data.");
        if (!isJitKernelSupportedByBuild(key))
        {
            return false;
        }

        constexpr int32_t kBEAM_SIZE{1};
        XQAKernelRuntimeHashKey const hashKey{trtToXqaDataType(key.dataType), trtToXqaDataType(key.kvDataType),
            key.headSize, key.specDecode ? 0 : key.qHeadsPerKv, kBEAM_SIZE, key.slidingWindow, key.tokensPerPage};

        std::lock_guard<std::mutex> lock(mMutex);
        auto const findIter = mFunctions.find(hashKey);
        if (findIter != mFunctions.end() && !findIter->second.empty())
        {
            // Reuse the loaded module when plugin clone or deserialization loads the same JIT key again.
            return true;
        }

        // Driver module loading requires a current CUDA context on this thread.
        CUDA_CHECK(cudaFree(nullptr));

        CUmodule hModule{};
        CUDA_DRIVER_CHECK(cuModuleLoadData(&hModule, cubinData));
        CudaModulePtr moduleGuard{hModule};

        XQAKernelFuncInfo funcInfo{};
        CUDA_DRIVER_CHECK(cuModuleGetFunction(&funcInfo.mDeviceFunction, moduleGuard.get(), "kernel_mha"));
        funcInfo.mHeadDim = static_cast<uint32_t>(key.headSize);
        funcInfo.mMTileSize = static_cast<uint32_t>(getXQAJitMTileSize(key));
        funcInfo.mSMVersion = static_cast<uint32_t>(key.sm);
        funcInfo.mKernelVariant = getJitKernelVariant(key);
        funcInfo.mRequiresClusterLaunch = jitRequiresClusterLaunch(key);
        funcInfo.mRequiresDistributedSharedMemory = funcInfo.mRequiresClusterLaunch;
        funcInfo.mSlidingWindow = key.slidingWindow;

        uint32_t* deviceSmemSize{nullptr};
        size_t dataSize{0};
        CUDA_DRIVER_CHECK(cuModuleGetGlobal(
            reinterpret_cast<CUdeviceptr*>(&deviceSmemSize), &dataSize, moduleGuard.get(), "smemSize"));
        CUDA_CHECK(cudaMemcpy(&funcInfo.mSharedMemBytes, deviceSmemSize, dataSize, cudaMemcpyDeviceToHost));

        XQADeviceCapability const deviceCapability = getDeviceCapability();
        if (!isKernelCompatibleWithDevice(funcInfo, deviceCapability))
        {
            return false;
        }

        if (funcInfo.mSharedMemBytes >= 46 * 1024)
        {
            CUDA_DRIVER_CHECK(cuFuncSetAttribute(
                funcInfo.mDeviceFunction, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, funcInfo.mSharedMemBytes));
        }

        mFunctions[hashKey].clear();
        mFunctions[hashKey].push_back(funcInfo);
        mJitModules.emplace_back(std::move(moduleGuard));
        return true;
    }

    XQAKernelFuncInfo findKernelFunction(XQAKernelRuntimeHashKey const& key) const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto const findIter = mFunctions.find(key);
        if (findIter == mFunctions.end() || findIter->second.empty())
        {
            // Return empty function info.
            return XQAKernelFuncInfo{};
        }

        auto const& candidates = findIter->second;
        auto const findBest = std::min_element(candidates.begin(), candidates.end(),
            [](XQAKernelFuncInfo const& lhs, XQAKernelFuncInfo const& rhs) noexcept {
                return getKernelVariantPriority(lhs.mKernelVariant) < getKernelVariantPriority(rhs.mKernelVariant);
            });
        return *findBest;
    }

protected:
    mutable std::mutex mMutex;

    std::vector<CudaModulePtr> mJitModules;

    std::unordered_map<XQAKernelRuntimeHashKey, std::vector<XQAKernelFuncInfo>, XQAKernelRuntimeHasher> mFunctions;
};

class XQAKernelLoader
{

public:
    //! @throws std::runtime_error if a CUDA driver error occurs
    XQAKernelList* getXQAKernelList(
        XQADataType dataType, XQADataType kvDataType, int32_t sm, bool specDecode, bool pagedKVCache, int32_t deviceId)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        XQAKernelLoadHashKey hash_key{dataType, kvDataType, sm, specDecode, pagedKVCache, deviceId};

        auto findIter = mKernels.find(hash_key);
        if (findIter == mKernels.end())
        {
            std::unique_ptr<XQAKernelList> newKernel = std::make_unique<XQAKernelList>();
            mKernels.insert(std::make_pair(hash_key, std::move(newKernel)));
            findIter = mKernels.find(hash_key);
        }
        return findIter->second.get();
    }

    static XQAKernelLoader& Get()
    {
        static XQAKernelLoader instance;
        return instance;
    }

private:
    XQAKernelLoader() = default;

    std::unordered_map<XQAKernelLoadHashKey, std::unique_ptr<XQAKernelList> const, XQAKernelLoadHasher> mKernels;
};

//! @throws std::runtime_error if a CUDA driver error occurs
inline XQAKernelList* getXQAKernels(
    XQADataType dataType, XQADataType kvDataType, int32_t sm, bool specDecode, bool pagedKVCache)
{
    int32_t deviceId = 0;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    return XQAKernelLoader::Get().getXQAKernelList(dataType, kvDataType, sm, specDecode, pagedKVCache, deviceId);
}

} // namespace

DecoderXQARunner::DecoderXQARunner(nvinfer1::DataType const dataType, nvinfer1::DataType const kvDataType,
    int32_t batchSize, int32_t numQHeads, int32_t numKvHeads, int32_t headSize, int32_t smVersion) noexcept
    : mDataType(dataType)
    , mKVDataType(kvDataType)
    , mBatchSize(batchSize)
    , mNumHeads(numQHeads)
    , mNumKVHeads(numKvHeads)
    , mHeadSize(headSize)
    , mSmVersion(smVersion)
{
}

XQALaunchParams DecoderXQARunner::initXQAParams() noexcept
{
    XQALaunchParams params{};
    params.numQheads = mNumHeads;
    params.numKVheads = mNumKVHeads;
    params.headSize = mHeadSize;
    params.batchSize = mBatchSize;
    params.dataType = mDataType;
    params.kvDataType = mKVDataType;
    params.headGroupSize = params.numQheads / params.numKVheads;

    return params;
}

bool DecoderXQARunner::loadDecodeXQAKernelFromCubin(XQAJitKey const& key, void const* cubinData, size_t cubinSize)
{
    XQAKernelList* xqaKernelList = getXQAKernels(trtToXqaDataType(key.dataType), trtToXqaDataType(key.kvDataType),
        key.sm, key.specDecode, key.tokensPerPage != 0);
    return xqaKernelList != nullptr && xqaKernelList->loadJitKernel(key, cubinData, cubinSize);
}

void DecoderXQARunner::dispatchXQAKernel(XQALaunchParams& params, cudaStream_t const& stream)
{
    // Check all device pointers are valid.
    check::check(params.output != nullptr && params.qInputPtr != nullptr && params.kvCache.data != nullptr
            && params.kvCache.sequence_lengths != nullptr
            && (!usePagedKVCache(params.kvCache) || params.kvCache.pageList != nullptr),
        "Invalid device pointer passed to kernel dispatch function");

    constexpr bool useSpecDecode = false;
    bool const usePagedKV = usePagedKVCache(params.kvCache);
    auto hashKey = getRuntimeHashKeyFromXQAParams(params);
    XQAKernelList* xqaKernelList = getXQAKernels(
        trtToXqaDataType(mDataType), trtToXqaDataType(mKVDataType), mSmVersion, useSpecDecode, usePagedKV);
    XQAKernelFuncInfo kernelInfo = xqaKernelList->findKernelFunction(hashKey);
    ELLM_CHECK(
        kernelInfo.mSharedMemBytes != 0, formatMissingXQAKernelMessage("GQA", hashKey, params, mSmVersion, false));

    KernelKVCacheArg kernelKVCacheArg = makeKernelKVCacheArg(params.kvCache, usePagedKV);
    void* const kvCacheArg = getKernelKVCacheArg(kernelKVCacheArg);

    auto kernelParams = makeXQAKernelParams(params, kernelInfo.mSlidingWindow, kvCacheArg);

    // The multi-block kernel launch is mainly for long sequence.
    // TODO: Add multiple block launch logic. The launch configuration highly depends on usecase and performance
    // context. Current measured workload doesn't get performance gain from multi-block launch.
    dim3 const dimGrid{
        getXQAKernelGridDimX(kernelInfo), static_cast<uint32_t>(mNumKVHeads), static_cast<uint32_t>(mBatchSize)};
    dim3 const dimCta = getXQAKernelCtaDim(kernelInfo);
    launchXQAKernel(kernelInfo, dimGrid, dimCta, stream, kernelParams.data());
}

void DecoderXQARunner::dispatchSpecDecodeXQAKernel(XQALaunchParams& params, cudaStream_t const& stream)
{
    // Check all device pointers are valid.
    check::check(params.output != nullptr && params.qInputPtr != nullptr && params.kvCache.data != nullptr
            && params.kvCache.sequence_lengths != nullptr && params.treeAttnMask != nullptr
            && (!usePagedKVCache(params.kvCache) || params.kvCache.pageList != nullptr),
        "Invalid device pointer passed to kernel dispatch function");

    constexpr bool useSpecDecode = true;
    bool const usePagedKV = usePagedKVCache(params.kvCache);
    auto hashKey = getRuntimeHashKeyFromXQAParamsSpecDecode(params);
    XQAKernelList* xqaKernelList = getXQAKernels(
        trtToXqaDataType(mDataType), trtToXqaDataType(mKVDataType), mSmVersion, useSpecDecode, usePagedKV);
    XQAKernelFuncInfo kernelInfo = xqaKernelList->findKernelFunction(hashKey);
    ELLM_CHECK(kernelInfo.mSharedMemBytes != 0,
        formatMissingXQAKernelMessage("Spec-DecodeGQA", hashKey, params, mSmVersion, true));

    KernelKVCacheArg kernelKVCacheArg = makeKernelKVCacheArg(params.kvCache, usePagedKV);
    void* const kvCacheArg = getKernelKVCacheArg(kernelKVCacheArg);

    auto kernelParams = makeSpecDecodeXQAKernelParams(params, kernelInfo.mSlidingWindow, kvCacheArg);
    int32_t const ctaTileY = static_cast<int32_t>(kernelInfo.mMTileSize);
    check::check(ctaTileY > 0, format::fmtstr("Invalid spec-decode ctaTileY %d in XQA kernel metadata.", ctaTileY));
    int32_t const tokenBlockPerGroup = (params.qSeqLen * params.headGroupSize - 1) / ctaTileY + 1;
    dim3 const dimGrid{getXQAKernelGridDimX(kernelInfo), static_cast<uint32_t>(mNumKVHeads * tokenBlockPerGroup),
        static_cast<uint32_t>(mBatchSize)};
    dim3 const dimCta = getXQAKernelCtaDim(kernelInfo);
    launchXQAKernel(kernelInfo, dimGrid, dimCta, stream, kernelParams.data());
}
