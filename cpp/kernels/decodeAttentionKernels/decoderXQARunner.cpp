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

#include "decoderXQARunner.h"
#include "common/checkMacros.h"
#include "cubin/xqa_kernel_cubin.h"

#include <algorithm>
#include <cuda.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;

using XQADataType = xqa::kernels::Data_type;

namespace
{

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
    int32_t sm;
    bool specDecode;

    bool operator==(XQAKernelLoadHashKey const& other) const
    {
        return data_type == other.data_type && sm == other.sm && specDecode == other.specDecode;
    }
};

struct XQAKernelLoadHasher
{
    size_t operator()(XQAKernelLoadHashKey const& s) const
    {
        size_t key = s.data_type;
        key <<= 16;
        key ^= s.sm;
        key <<= 4;
        key ^= s.specDecode;
        return key;
    }
};

struct XQAKernelRuntimeHashKey
{
    XQADataType kv_data_type;
    int32_t head_size;
    int32_t num_q_heads_per_kv;
    int32_t beam_size;

    bool operator==(XQAKernelRuntimeHashKey const& other) const
    {
        return kv_data_type == other.kv_data_type && head_size == other.head_size
            && num_q_heads_per_kv == other.num_q_heads_per_kv && beam_size == other.beam_size;
    }
};

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParams(XQALaunchParams const& xqaParams)
{
    constexpr int32_t kBEAM_SIZE{1};
    int32_t numQHeadPerKV = xqaParams.numQheads / xqaParams.numKVheads;
    return {trtToXqaDataType(xqaParams.dataType), xqaParams.headSize, numQHeadPerKV, kBEAM_SIZE};
}

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParamsSpecDecode(XQALaunchParams const& xqaParams)
{
    constexpr int32_t kBEAM_SIZE{1};
    constexpr int32_t kQHEAD_PER_KV = 0; // Tree attention kernel supports any ratio of Q/KV heads.
    return {trtToXqaDataType(xqaParams.dataType), xqaParams.headSize, kQHEAD_PER_KV, kBEAM_SIZE};
}

struct XQAKernelRuntimeHasher
{
    size_t operator()(XQAKernelRuntimeHashKey const& s) const
    {
        size_t key = s.kv_data_type;
        key <<= 16;
        key ^= s.head_size;
        key <<= 8;
        key ^= s.num_q_heads_per_kv;
        key <<= 8;
        key ^= s.beam_size;
        return key;
    }
};

struct XQAKernelFuncInfo
{
    uint32_t mSharedMemBytes{0};
    CUfunction mDeviceFunction{0};
};

class XQAKernelList
{
    using TKernelMetaInfo = xqa::kernels::XQAKernelMetaInfo;

public:
    XQAKernelList(XQADataType type, int32_t sm, bool specDecode)
        : mKernelMeta(nullptr)
        , mKernelMetaCount(0)
        , mSMVersion(sm)
        , mDataType(type)
        , mSpecDecode(specDecode)
    {
        mKernelMeta = &(xqa::kernels::sXqaKernelMetaInfo[0]);
        mKernelMetaCount = sizeof(xqa::kernels::sXqaKernelMetaInfo) / sizeof(xqa::kernels::sXqaKernelMetaInfo[0]);
    }

    void loadXQAKernels()
    {
        if (!mFunctions.empty())
        {
            return;
        }
        for (int32_t i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            if (kernelMeta.mDataType != mDataType || kernelMeta.mSM != mSMVersion || kernelMeta.mCubin == nullptr)
            {
                continue;
            }
            // Filter out kernel that irrelevant to this project.
            if (kernelMeta.mPagedKVCache == true || kernelMeta.mBeamWidth != 1
                || kernelMeta.mDataType != kernelMeta.mKVDataType)
            {
                continue;
            }
            if (kernelMeta.mMultiQueryTokens != mSpecDecode)
            {
                continue;
            }
            // load CUmodule
            CUmodule hModule;
            auto findModuleIter = mModules.find(kernelMeta.mCubin);
            if (findModuleIter != mModules.end())
            {
                hModule = findModuleIter->second;
            }
            else
            {
                CUDA_DRIVER_CHECK(cuModuleLoadData(&hModule, kernelMeta.mCubin));
                mModules.insert(std::make_pair(kernelMeta.mCubin, hModule));
            }

            XQAKernelFuncInfo funcInfo{};
            CUDA_DRIVER_CHECK(cuModuleGetFunction(&funcInfo.mDeviceFunction, hModule, kernelMeta.mFuncName));

            uint32_t* deviceSmemSize{nullptr};
            size_t dataSize{0};
            CUDA_DRIVER_CHECK(
                cuModuleGetGlobal(reinterpret_cast<CUdeviceptr*>(&deviceSmemSize), &dataSize, hModule, "smemSize"));
            // Use of default stream is inevitable and justified here because it is called during kernel loading phase,
            // not runtime.
            CUDA_CHECK(cudaMemcpy(&funcInfo.mSharedMemBytes, deviceSmemSize, dataSize, cudaMemcpyDeviceToHost));

            // Set 46KB threshold here because we have to take static/driver shared memory into consideration.
            // Default value for shared memory is 48KB.
            if (funcInfo.mSharedMemBytes >= 46 * 1024)
            {
                CUDA_DRIVER_CHECK(cuFuncSetAttribute(funcInfo.mDeviceFunction,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, funcInfo.mSharedMemBytes));
            }
            XQAKernelRuntimeHashKey hashKey{
                kernelMeta.mKVDataType, kernelMeta.mHeadDim, kernelMeta.mNumQHeadsOverKV, kernelMeta.mBeamWidth};
            mFunctions.insert(std::make_pair(hashKey, funcInfo));
        }
    }

    XQAKernelFuncInfo findKernelFunction(XQAKernelRuntimeHashKey const& key) const
    {
        auto const findIter = mFunctions.find(key);
        if (findIter == mFunctions.end())
        {
            // Return empty function info.
            return XQAKernelFuncInfo{};
        }

        return findIter->second;
    }

protected:
    TKernelMetaInfo const* mKernelMeta;
    int32_t mKernelMetaCount;
    int32_t mSMVersion;
    bool mSpecDecode;
    XQADataType mDataType;
    std::unordered_map<unsigned long long const*, CUmodule> mModules;

    std::unordered_map<XQAKernelRuntimeHashKey, XQAKernelFuncInfo, XQAKernelRuntimeHasher> mFunctions;
};

class XQAKernelLoader
{

public:
    XQAKernelList* getXQAKernelList(XQADataType type, int32_t sm, bool specDecode)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        XQAKernelLoadHashKey hash_key{type, sm, specDecode};

        auto findIter = mKernels.find(hash_key);
        if (findIter == mKernels.end())
        {
            std::unique_ptr<XQAKernelList> newKernel = std::make_unique<XQAKernelList>(type, sm, specDecode);
            newKernel->loadXQAKernels();
            mKernels.insert(std::make_pair(hash_key, std::move(newKernel)));
            findIter = mKernels.find(hash_key);
        }
        return findIter->second.get();
    }

    static XQAKernelLoader& Get()
    {
        static std::unique_ptr<XQAKernelLoader> kernelLoader = nullptr;
        if (kernelLoader == nullptr)
        {
            kernelLoader = std::make_unique<XQAKernelLoader>(XQAKernelLoader());
        }

        return *kernelLoader;
    }

private:
    XQAKernelLoader() = default;

    std::unordered_map<XQAKernelLoadHashKey, std::unique_ptr<XQAKernelList> const, XQAKernelLoadHasher> mKernels;
};

inline XQAKernelList* getXQAKernels(XQADataType type, int32_t sm, bool specDecode)
{
    return XQAKernelLoader::Get().getXQAKernelList(type, sm, specDecode);
}

} // namespace

DecoderXQARunner::DecoderXQARunner(nvinfer1::DataType const dataType, int32_t batchSize, int32_t numQHeads,
    int32_t numKvHeads, int32_t headSize, int32_t smVersion)
    : mDataType(dataType)
    , mBatchSize(batchSize)
    , mNumHeads(numQHeads)
    , mNumKVHeads(numKvHeads)
    , mHeadSize(headSize)
    , mSmVersion(smVersion)
{
}

XQALaunchParams DecoderXQARunner::initXQAParams()
{
    XQALaunchParams params{};
    params.numQheads = mNumHeads;
    params.numKVheads = mNumKVHeads;
    params.headSize = mHeadSize;
    params.batchSize = mBatchSize;
    params.dataType = mDataType;
    params.headGroupSize = params.numQheads / params.numKVheads;

    return params;
}

bool DecoderXQARunner::canImplement(int32_t numQHeads, int32_t numKVHeads, int32_t smVersion, DataType dataType)
{
    bool const checkHeadNumbers = numQHeads % numKVHeads == 0;
    bool const checkType = dataType == DataType::kHALF;
    std::vector<int32_t> allowedSMVersions{80, 86, 87, 89, 100, 101, 120, 121};
    bool const checkSMVersion
        = std::find(allowedSMVersions.begin(), allowedSMVersions.end(), smVersion) != allowedSMVersions.end();

    // Current kernel list supports head ratio from 1 to 8.
    int32_t const headRatio = numQHeads / numKVHeads;
    bool const checkQHeadPerKV = headRatio >= 1 && headRatio <= 8;

    return checkHeadNumbers && checkType && checkSMVersion && checkQHeadPerKV;
}

bool DecoderXQARunner::loadDecodeXQAKernels(int32_t smVersion, DataType dataType, bool useSpecDecodeKernels)
{
    XQAKernelList* xqaKernelList = getXQAKernels(trtToXqaDataType(dataType), smVersion, useSpecDecodeKernels);
    return xqaKernelList != nullptr;
}

void DecoderXQARunner::dispatchXQAKernel(XQALaunchParams& params, cudaStream_t const& stream)
{
    // Check all device pointers are valid.
    check::check(params.output != nullptr && params.qInputPtr != nullptr && params.kvCache.data != nullptr
            && params.kvCache.sequence_lengths != nullptr,
        "Invalid device pointer passed to kernel dispatch function");

    constexpr bool useSpecDecode = false;
    auto hashKey = getRuntimeHashKeyFromXQAParams(params);
    XQAKernelList* xqaKernelList = getXQAKernels(trtToXqaDataType(mDataType), mSmVersion, useSpecDecode);
    XQAKernelFuncInfo kernelInfo = xqaKernelList->findKernelFunction(hashKey);
    check::check(kernelInfo.mSharedMemBytes != 0, "No available kernel available for the GQA");

    void* kernelParams[]
        = {&params.numKVheads, &params.qScale, &params.output, &params.qInputPtr, &params.attentionSinks,
            &params.kvCache, &params.batchSize, &params.kvScale, &params.semaphores, &params.scratch};

    // The multi-block kernel launch is mainly for long sequence.
    // TODO: Add multiple block launch logic. The launch configuration highly depends on usecase and performance
    // context. Current measured workload doesn't get performance gain from multi-block launch.
    dim3 const dimGrid{1, mNumKVHeads, mBatchSize};
    dim3 const dimCta{128, 1, 2};
    CUDA_DRIVER_CHECK(cuLaunchKernel(kernelInfo.mDeviceFunction, dimGrid.x, dimGrid.y, dimGrid.z, dimCta.x, dimCta.y,
        dimCta.z, kernelInfo.mSharedMemBytes, stream, kernelParams, nullptr));
}

void DecoderXQARunner::dispatchSpecDecodeXQAKernel(XQALaunchParams& params, cudaStream_t const& stream)
{
    // Check all device pointers are valid.
    check::check(params.output != nullptr && params.qInputPtr != nullptr && params.kvCache.data != nullptr
            && params.kvCache.sequence_lengths != nullptr && params.treeAttnMask != nullptr,
        "Invalid device pointer passed to kernel dispatch function");

    constexpr bool useSpecDecode = true;
    auto hashKey = getRuntimeHashKeyFromXQAParamsSpecDecode(params);
    XQAKernelList* xqaKernelList = getXQAKernels(trtToXqaDataType(mDataType), mSmVersion, useSpecDecode);
    XQAKernelFuncInfo kernelInfo = xqaKernelList->findKernelFunction(hashKey);
    check::check(kernelInfo.mSharedMemBytes != 0, "No available kernel available for the Spec-DecodeGQA");

    void* kernelParams[] = {&params.qSeqLen, &params.numKVheads, &params.headGroupSize, &params.qCuSeqLen,
        &params.qScale, &params.output, &params.qInputPtr, &params.treeAttnMask, &params.attentionSinks,
        &params.kvCache, &params.batchSize, &params.kvScale, &params.semaphores, &params.scratch};
    constexpr int32_t CTA_TILE_Y = 32;
    int32_t const tokenBlockPerGroup = (params.qSeqLen * params.headGroupSize - 1) / CTA_TILE_Y + 1;
    dim3 const dimGrid{1, mNumKVHeads * tokenBlockPerGroup, mBatchSize};
    dim3 const dimCta{128, 1, 2};
    CUDA_DRIVER_CHECK(cuLaunchKernel(kernelInfo.mDeviceFunction, dimGrid.x, dimGrid.y, dimGrid.z, dimCta.x, dimCta.y,
        dimCta.z, kernelInfo.mSharedMemBytes, stream, kernelParams, nullptr));
}
