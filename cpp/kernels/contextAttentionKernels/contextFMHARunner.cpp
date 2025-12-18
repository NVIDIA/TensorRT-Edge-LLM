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

#include "contextFMHARunner.h"
#include "common/checkMacros.h"
#include "cubin/fmha_cubin.h"
#include "fmhaParams_v2.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <memory>
#include <mutex>
#include <unordered_map>

using namespace nvinfer1;
using namespace trt_edgellm;

using FMHADataType = fmha_v2::Data_type;

namespace
{
union __half2_uint32_t_union
{
    half2 fp162;
    uint32_t u32;
};

union __float_uint32_t_union
{
    float fp32;
    uint32_t u32;
};

static inline void set_alpha(uint32_t& alpha, float norm, FMHADataType dtype)
{
    if (dtype == FMHADataType::DATA_TYPE_FP16)
    {
        // Convert the float value into two fp16 value and pack into the uint32_t buffer.
        __half2_uint32_t_union temp;
        temp.fp162 = __float2half2_rn(norm);
        alpha = temp.u32;
    }
    else if (dtype == FMHADataType::DATA_TYPE_FP32)
    {
        __float_uint32_t_union temp;
        temp.fp32 = norm;
        alpha = temp.u32;
    }
    else if (dtype == FMHADataType::DATA_TYPE_INT32)
    {
        int32_t inorm = static_cast<int32_t>(norm);
        alpha = reinterpret_cast<uint32_t const&>(inorm);
    }
    else if (dtype == FMHADataType::DATA_TYPE_BF16)
    {
        // TODO HACK!! BF16 Outputs are computed in FP32 for FP8.
        // This is because cublas does not allow current FP32 output.
        alpha = reinterpret_cast<uint32_t const&>(norm);
    }
    else
    {
        check::check(false, "Unsupported type for alpha value");
    }
}

FMHADataType trtToFMHADataType(nvinfer1::DataType type)
{
    FMHADataType fmhaType{FMHADataType::DATA_TYPE_FP16};
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: fmhaType = FMHADataType::DATA_TYPE_FP32; break;
    case nvinfer1::DataType::kHALF: fmhaType = FMHADataType::DATA_TYPE_FP16; break;
    case nvinfer1::DataType::kBF16: fmhaType = FMHADataType::DATA_TYPE_BF16; break;
    case nvinfer1::DataType::kFP8: fmhaType = FMHADataType::DATA_TYPE_E4M3; break;
    default: throw std::runtime_error("Unsupported datatype for FMHA_v2.");
    }
    return fmhaType;
}

int32_t attentionMaskTypeToInt(ContextAttentionMaskType type)
{
    int32_t result{};
    switch (type)
    {
    case ContextAttentionMaskType::PADDING: result = 0; break;
    case ContextAttentionMaskType::CAUSAL: result = 1; break;
    case ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL: result = 2; break;
    }
    return result;
}

int32_t attentionInputLayoutToInt(AttentionInputLayout layout)
{
    int32_t result{};
    switch (layout)
    {
    case AttentionInputLayout::PACKED_QKV: result = 0; break;
    case AttentionInputLayout::CONTIGUOUS_Q_KV: result = 1; break;
    case AttentionInputLayout::Q_PAGED_KV: result = 2; break;
    case AttentionInputLayout::SEPARATE_Q_K_V: result = 3; break;
    }
    return result;
}

struct FMHAKernelLoadHashKey
{
    FMHADataType data_type;
    int32_t sm;

    bool operator==(FMHAKernelLoadHashKey const& other) const
    {
        return data_type == other.data_type && sm == other.sm;
    }
};

struct FMHAKernelLoadHasher
{
    size_t operator()(FMHAKernelLoadHashKey const& s) const
    {
        size_t key = s.data_type;
        key <<= 16;
        key ^= s.sm;
        return key;
    }
};

struct FMHAKernelHashKey
{
    FMHADataType data_type;
    int32_t sequenceLen;
    int32_t headSize;
    bool unroll;
    bool force_fp32_acc;
    bool flash_attention;
    int32_t attention_mask_type;
    bool tiled;
    int32_t attention_input_layout;

    bool operator==(FMHAKernelHashKey const& other) const
    {
        // Flash attention kernel supports any sequence length. So for this set of kernel, we will match any sequence
        // length.
        return data_type == other.data_type && (sequenceLen == other.sequenceLen || flash_attention == true)
            && headSize == other.headSize && unroll == other.unroll && force_fp32_acc == other.force_fp32_acc
            && flash_attention == other.flash_attention && attention_mask_type && other.attention_mask_type
            && tiled == other.tiled && attention_input_layout == other.attention_input_layout;
    }
};

struct FMHAKernelHasher
{
    size_t operator()(FMHAKernelHashKey const& hashKey) const
    {
        // flash attention support unlimited-sequence length
        int32_t s = hashKey.flash_attention ? 0 : hashKey.sequenceLen;
        // D <= 2048
        return (size_t) s << 32 | hashKey.headSize << 16 | (hashKey.attention_mask_type << 6)
            | (hashKey.tiled ? 16ull : 0ull) | (hashKey.force_fp32_acc ? 8ull : 0ull)
            | (hashKey.flash_attention ? 4ull : 0ull) | (hashKey.unroll ? 2ull : 0ull);
    }
};

struct FMHAKernelFuncInfo
{
    uint32_t mThreadsPerCTA;
    uint32_t mUnrollStep;
    uint32_t mSharedMemBytes{0};
    CUfunction mDeviceFunction{0};
    std::string mFuncName{};
};

class FMHAKernelList
{
    using TKernelMetaInfo = fmha_v2::FusedMultiHeadAttentionKernelMetaInfoV2;

public:
    FMHAKernelList(FMHADataType type, int32_t sm)
        : mDataType(type)
        , mSMVersion(sm)
    {
        mKernelMeta = &(fmha_v2::sMhaKernelMetaInfosV2[0]);
        mKernelMetaCount = sizeof(fmha_v2::sMhaKernelMetaInfosV2) / sizeof(fmha_v2::sMhaKernelMetaInfosV2[0]);
    }

    void loadFMHAKernels()
    {
        if (!mFunctions.empty())
        {
            return;
        }
        for (int32_t i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            if (kernelMeta.mDataTypeIn != mDataType || kernelMeta.mDataTypeOut != mDataType
                || kernelMeta.mSM != mSMVersion || kernelMeta.mCubin == nullptr)
            {
                continue;
            }

            // load CUmodule. Each module can contain multiple kernel function.
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

            FMHAKernelFuncInfo funcInfo{};
            CUDA_DRIVER_CHECK(cuModuleGetFunction(&funcInfo.mDeviceFunction, hModule, kernelMeta.mFuncName));
            funcInfo.mSharedMemBytes = kernelMeta.mSharedMemBytes;
            funcInfo.mThreadsPerCTA = kernelMeta.mThreadsPerCTA;
            funcInfo.mUnrollStep = kernelMeta.mUnrollStep;
            funcInfo.mFuncName = std::string(kernelMeta.mFuncName);

            if (funcInfo.mSharedMemBytes >= 48 * 1024)
            {
                CUDA_DRIVER_CHECK(cuFuncSetAttribute(funcInfo.mDeviceFunction,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, funcInfo.mSharedMemBytes));
            }
            FMHAKernelHashKey hashKey{kernelMeta.mDataTypeIn, static_cast<int32_t>(kernelMeta.mS),
                static_cast<int32_t>(kernelMeta.mD), kernelMeta.mUnrollStep != 0, kernelMeta.mFP32Accumulation,
                kernelMeta.mFlashAttention, kernelMeta.mAttentionMaskType, kernelMeta.mTiled,
                kernelMeta.mAttentionInputLayout};
            mFunctions.insert(std::make_pair(hashKey, funcInfo));
        }
    }

    FMHAKernelFuncInfo findKernelFunction(FMHAKernelHashKey const& key) const
    {
        auto const findIter = mFunctions.find(key);
        if (findIter == mFunctions.end())
        {
            // Return empty function info.
            return FMHAKernelFuncInfo{};
        }

        return findIter->second;
    }

protected:
    TKernelMetaInfo const* mKernelMeta;
    int32_t mKernelMetaCount;
    FMHADataType mDataType;
    uint32_t mSMVersion;
    std::unordered_map<unsigned char const*, CUmodule> mModules;

    std::unordered_map<FMHAKernelHashKey, FMHAKernelFuncInfo, FMHAKernelHasher> mFunctions;
};

class FMHAKernelLoader
{

public:
    FMHAKernelList* getFMHAKernelList(FMHADataType type, int32_t sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        FMHAKernelLoadHashKey hash_key{type, sm};

        auto findIter = mKernels.find(hash_key);
        if (findIter == mKernels.end())
        {
            std::unique_ptr<FMHAKernelList> newKernel = std::make_unique<FMHAKernelList>(type, sm);
            newKernel->loadFMHAKernels();
            mKernels.insert(std::make_pair(hash_key, std::move(newKernel)));
            findIter = mKernels.find(hash_key);
        }
        return findIter->second.get();
    }

    static FMHAKernelLoader& Get()
    {
        static std::unique_ptr<FMHAKernelLoader> kernelLoader = nullptr;
        if (kernelLoader == nullptr)
        {
            kernelLoader = std::make_unique<FMHAKernelLoader>(FMHAKernelLoader());
        }

        return *kernelLoader;
    }

private:
    FMHAKernelLoader() = default;

    std::unordered_map<FMHAKernelLoadHashKey, std::unique_ptr<FMHAKernelList> const, FMHAKernelLoadHasher> mKernels;
};

inline FMHAKernelList* getFMHAKernels(FMHADataType type, int32_t sm)
{
    return FMHAKernelLoader::Get().getFMHAKernelList(type, sm);
}

}; // namespace

ContextFMHARunner::ContextFMHARunner(nvinfer1::DataType const dataType, int32_t batchSize, int32_t paddedSeqLen,
    int32_t numQHeads, int32_t numKvHeads, int32_t headSize, int32_t smVersion, AttentionInputLayout inputLayout)
    : mDataType(dataType)
    , mBatchSize(batchSize)
    , mPaddedSequenceLen(paddedSeqLen)
    , mNumHeads(numQHeads)
    , mNumKVHeads(numKvHeads)
    , mHeadSize(headSize)
    , mSmVersion(smVersion)
{
    // The context FMHA-v2 kernels taken by the project only support ampere/ada for
    // reference on x86 machine, Orin/Thor for production on auto platforms.
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    mLaunchParams.multi_processor_count = props.multiProcessorCount;
    mLaunchParams.device_l2_cache_size = props.l2CacheSize;
    // Only causal attention kernel get integrated and used now.
    mLaunchParams.attention_mask_type = ContextAttentionMaskType::CAUSAL;
    mLaunchParams.attention_input_layout = inputLayout;

    bool const isSm8x = (smVersion == fmha_v2::kSM_80 || smVersion == fmha_v2::kSM_86 || smVersion == fmha_v2::kSM_87
        || smVersion == fmha_v2::kSM_89);
    bool const isSm10x = (smVersion == fmha_v2::kSM_100 || smVersion == fmha_v2::kSM_101);
    bool const isSm12x = (smVersion == fmha_v2::kSM_120 || smVersion == fmha_v2::kSM_121);
    check::check((isSm8x || isSm10x || isSm12x), "Other SMs are not supported by context FMHA-v2 kernels");
    // Handle kernel selection under different context.
    if (isSm8x || isSm10x || isSm12x)
    {
        // always use flash attention kernels for Ampere/Ada
        mLaunchParams.flash_attention = true;
        // flash attention kernles s = 0 (support any seq length)
        mLaunchParams.force_unroll = true;

        // TODO: Check if still proper for contiguous q-kv input layout
        if (mPaddedSequenceLen <= 64 || mHeadSize < 256)
        {
            // flash attention tiled kernels allows larger free dim tile size (M, N) with flexibility
            // in unroll dimension tile size (K). for short sequence length (s<=128), tiled kernels
            // can suffer from tile quantization loss.
            // Also flash attention tiled kernel is generally faster when head_size>=256
            mLaunchParams.use_granular_tiling = false;
        }
        else
        {
            // otherwise, choose tiled FMHA-v2 flash-attention kernel.
            mLaunchParams.use_granular_tiling = true;
        }
    }
}

void ContextFMHARunner::setupParams(FusedMultiheadAttentionParamsV2& params)
{
    float const invSqrtScale = (1.f / sqrtf(mHeadSize));

    float const scale_bmm1 = invSqrtScale;
    float const scale_softmax = 1.f; // Seems to be only required for int8
    float const scale_bmm2 = 1.f;

    FMHADataType scale_type = mLaunchParams.force_fp32_acc ? fmha_v2::DATA_TYPE_FP32 : trtToFMHADataType(mDataType);
    set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
    set_alpha(params.scale_softmax, scale_softmax, scale_type);
    set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

    params.b = mBatchSize;
    params.h = mNumHeads;
    params.h_kv = mNumKVHeads;
    params.h_q_per_kv = mNumHeads / mNumKVHeads;
    params.s = mPaddedSequenceLen; // max sequence length of a batch of input queries.
    params.d = mHeadSize;
    params.dv = mHeadSize;
    params.is_s_padded = true;

    params.o_stride_in_bytes = mNumHeads * mHeadSize * sizeof(half);

    check::check(mLaunchParams.attention_input_layout == AttentionInputLayout::PACKED_QKV
            || mLaunchParams.attention_input_layout == AttentionInputLayout::CONTIGUOUS_Q_KV,
        "Unsupported input layout");
    if (mLaunchParams.attention_input_layout == AttentionInputLayout::PACKED_QKV)
    {
        int64_t stride_in_bytes = (mNumHeads + 2 * mNumKVHeads) * mHeadSize * sizeof(half);
        params.q_stride_in_bytes = stride_in_bytes;
        params.k_stride_in_bytes = stride_in_bytes;
        params.v_stride_in_bytes = stride_in_bytes;
    }
    else
    {
        int64_t q_stride_in_bytes = mNumHeads * mHeadSize * sizeof(half);
        int64_t kv_stride_in_bytes = (2 * mNumKVHeads) * mHeadSize * sizeof(half);
        params.q_stride_in_bytes = q_stride_in_bytes;
        params.k_stride_in_bytes = kv_stride_in_bytes;
        params.v_stride_in_bytes = kv_stride_in_bytes;
    }
}

bool ContextFMHARunner::canImplement(int32_t headSize, [[maybe_unused]] int32_t sm, nvinfer1::DataType dataType)
{
    bool const checkType = dataType == DataType::kHALF;
    bool const checkHeadSize = headSize == 128 || headSize == 64;

    return checkType && checkHeadSize;
}

bool ContextFMHARunner::loadContextFMHAKernels(int32_t smVersion, nvinfer1::DataType dataType)
{
    FMHAKernelList* fmhaKernelList = getFMHAKernels(trtToFMHADataType(dataType), smVersion);
    return fmhaKernelList != nullptr;
}

void ContextFMHARunner::dispatchFMHAKernel(FusedMultiheadAttentionParamsV2& params, cudaStream_t const& stream)
{
    if (mLaunchParams.attention_input_layout == AttentionInputLayout::PACKED_QKV)
    {
        check::check(params.qkv_ptr != nullptr && params.o_ptr != nullptr && params.cu_q_seqlens != nullptr,
            "Device pointers are supposed to be valid");
    }
    else // CONTIGUOUS_Q_KV
    {
        check::check(params.q_ptr != nullptr && params.kv_ptr != nullptr && params.o_ptr != nullptr
                && params.cu_q_seqlens != nullptr && params.cu_kv_seqlens != nullptr,
            "Device pointers are supposed to be valid");
    }
    FMHAKernelHashKey hashKey{trtToFMHADataType(mDataType), mPaddedSequenceLen, mHeadSize, mLaunchParams.force_unroll,
        mLaunchParams.force_fp32_acc, mLaunchParams.flash_attention,
        attentionMaskTypeToInt(mLaunchParams.attention_mask_type), mLaunchParams.use_granular_tiling,
        attentionInputLayoutToInt(mLaunchParams.attention_input_layout)};
    FMHAKernelList* fmhaKernelList = getFMHAKernels(trtToFMHADataType(mDataType), mSmVersion);
    FMHAKernelFuncInfo kernelInfo = fmhaKernelList->findKernelFunction(hashKey);
    check::check(kernelInfo.mSharedMemBytes != 0, "There must be one kernel to implement the MHA");

    void* kernelParams[] = {&params, nullptr};
    // Right now we onlu use flash attention kernel
    // flash attention supports any sequence length (0 in kernel meta)
    int32_t unroll = (params.s + kernelInfo.mUnrollStep - 1) / kernelInfo.mUnrollStep;
    // on Ampere/Ada flash attention, we launch blocks (steps, h, b)
    // TODO: Generalize the logic for more architectures.
    CUDA_DRIVER_CHECK(cuLaunchKernel(kernelInfo.mDeviceFunction, unroll, params.h, params.b, kernelInfo.mThreadsPerCTA,
        1, 1, kernelInfo.mSharedMemBytes, stream, kernelParams, nullptr));
}
