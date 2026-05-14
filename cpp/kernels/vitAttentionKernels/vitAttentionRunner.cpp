/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vitAttentionRunner.h"

#include "common/checkMacros.h"
#include "cubin/vit_fmha_cubin.h"
#include "kernels/contextAttentionKernels/fmhaParams_v2.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace trt_edgellm
{
namespace
{

using FMHADataType = vit_fmha::Data_type;

union Half2Uint32Union
{
    half2 fp162;
    uint32_t u32;
};

void setAlpha(uint32_t& alpha, float norm)
{
    Half2Uint32Union temp;
    temp.fp162 = __float2half2_rn(norm);
    alpha = temp.u32;
}

FMHADataType trtToFMHADataType(nvinfer1::DataType type)
{
    check::check(type == nvinfer1::DataType::kHALF, "ViT FMHA only supports FP16.");
    return FMHADataType::DATA_TYPE_FP16;
}

int32_t getCurrentSMVersion()
{
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    int32_t smVersion = props.major * 10 + props.minor;
    // Workaround for CUDA12/13 Thor re-numbering. The kernels themselves have version compatibility.
    if (smVersion == 110)
    {
        smVersion = 101;
    }
    return smVersion;
}

struct FMHAKernelLoadHashKey
{
    FMHADataType dataType;
    int32_t sm;

    bool operator==(FMHAKernelLoadHashKey const& other) const
    {
        return dataType == other.dataType && sm == other.sm;
    }
};

struct FMHAKernelLoadHasher
{
    size_t operator()(FMHAKernelLoadHashKey const& key) const
    {
        return (static_cast<size_t>(key.dataType) << 16) ^ static_cast<size_t>(key.sm);
    }
};

struct FMHAKernelHashKey
{
    FMHADataType dataType;
    int32_t sequenceLen;
    int32_t headSize;
    bool unroll;
    bool forceFP32Acc;
    bool flashAttention;
    int32_t attentionMaskType;
    bool tiled;
    int32_t attentionInputLayout;

    bool operator==(FMHAKernelHashKey const& other) const
    {
        return dataType == other.dataType && (sequenceLen == other.sequenceLen || flashAttention)
            && headSize == other.headSize && unroll == other.unroll && forceFP32Acc == other.forceFP32Acc
            && flashAttention == other.flashAttention && attentionMaskType == other.attentionMaskType
            && tiled == other.tiled && attentionInputLayout == other.attentionInputLayout;
    }
};

struct FMHAKernelHasher
{
    size_t operator()(FMHAKernelHashKey const& key) const
    {
        int32_t const s = key.flashAttention ? 0 : key.sequenceLen;
        return (static_cast<size_t>(s) << 32) | (static_cast<size_t>(key.headSize) << 16)
            | (static_cast<size_t>(key.attentionMaskType) << 6) | (key.tiled ? 16ull : 0ull)
            | (key.forceFP32Acc ? 8ull : 0ull) | (key.flashAttention ? 4ull : 0ull)
            | (key.unroll ? 2ull : 0ull) | static_cast<size_t>(key.attentionInputLayout);
    }
};

std::string describeFMHAKernelKey(FMHAKernelHashKey const& key, int32_t smVersion)
{
    std::ostringstream message;
    message << "There must be one ViT FMHA kernel to implement the MHA. Requested sm=" << smVersion
            << ", dtype=" << static_cast<int32_t>(key.dataType) << ", head_size=" << key.headSize
            << ", sequence_len=" << key.sequenceLen << ", unroll=" << key.unroll
            << ", fp32_acc=" << key.forceFP32Acc << ", flash=" << key.flashAttention
            << ", mask_type=" << key.attentionMaskType << ", tiled=" << key.tiled
            << ", input_layout=" << key.attentionInputLayout << ".";
    return message.str();
}

struct FMHAKernelFuncInfo
{
    uint32_t threadsPerCTA{};
    uint32_t unrollStep{};
    uint32_t sharedMemBytes{};
    CUfunction deviceFunction{};
};

class ViTFMHAKernelList
{
public:
    ViTFMHAKernelList(FMHADataType type, int32_t sm)
        : mDataType(type)
        , mSMVersion(sm)
    {
    }

    void load()
    {
        if (!mFunctions.empty())
        {
            return;
        }

        auto const* kernelMeta = &(vit_fmha::sMhaKernelMetaInfosV2[0]);
        int32_t const kernelMetaCount
            = sizeof(vit_fmha::sMhaKernelMetaInfosV2) / sizeof(vit_fmha::sMhaKernelMetaInfosV2[0]);
        for (int32_t i = 0; i < kernelMetaCount; ++i)
        {
            auto const& meta = kernelMeta[i];
            if (meta.mDataTypeIn != mDataType || meta.mDataTypeOut != mDataType || meta.mSM != mSMVersion
                || meta.mCubin == nullptr)
            {
                continue;
            }

            CUmodule module{};
            auto moduleIter = mModules.find(meta.mCubin);
            if (moduleIter != mModules.end())
            {
                module = moduleIter->second;
            }
            else
            {
                CUDA_DRIVER_CHECK(cuModuleLoadData(&module, meta.mCubin));
                mModules.insert(std::make_pair(meta.mCubin, module));
            }

            FMHAKernelFuncInfo funcInfo{};
            CUDA_DRIVER_CHECK(cuModuleGetFunction(&funcInfo.deviceFunction, module, meta.mFuncName));
            funcInfo.sharedMemBytes = meta.mSharedMemBytes;
            funcInfo.threadsPerCTA = meta.mThreadsPerCTA;
            funcInfo.unrollStep = meta.mUnrollStep;

            if (funcInfo.sharedMemBytes >= 48 * 1024)
            {
                CUDA_DRIVER_CHECK(cuFuncSetAttribute(funcInfo.deviceFunction,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, funcInfo.sharedMemBytes));
            }

            FMHAKernelHashKey key{meta.mDataTypeIn, static_cast<int32_t>(meta.mS), static_cast<int32_t>(meta.mD),
                meta.mUnrollStep != 0, meta.mFP32Accumulation, meta.mFlashAttention, meta.mAttentionMaskType,
                meta.mTiled, meta.mAttentionInputLayout};
            mFunctions.insert(std::make_pair(key, funcInfo));
        }
    }

    FMHAKernelFuncInfo find(FMHAKernelHashKey const& key) const
    {
        auto iter = mFunctions.find(key);
        if (iter == mFunctions.end())
        {
            return FMHAKernelFuncInfo{};
        }
        return iter->second;
    }

private:
    FMHADataType mDataType;
    int32_t mSMVersion;
    std::unordered_map<unsigned char const*, CUmodule> mModules;
    std::unordered_map<FMHAKernelHashKey, FMHAKernelFuncInfo, FMHAKernelHasher> mFunctions;
};

class ViTFMHAKernelLoader
{
public:
    ViTFMHAKernelList* get(FMHADataType type, int32_t sm)
    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        FMHAKernelLoadHashKey key{type, sm};
        auto iter = mKernels.find(key);
        if (iter == mKernels.end())
        {
            auto kernelList = std::make_unique<ViTFMHAKernelList>(type, sm);
            kernelList->load();
            mKernels.insert(std::make_pair(key, std::move(kernelList)));
            iter = mKernels.find(key);
        }
        return iter->second.get();
    }

    static ViTFMHAKernelLoader& instance()
    {
        static ViTFMHAKernelLoader loader;
        return loader;
    }

private:
    std::unordered_map<FMHAKernelLoadHashKey, std::unique_ptr<ViTFMHAKernelList>, FMHAKernelLoadHasher> mKernels;
};

void setupFMHAParams(FusedMultiheadAttentionParamsV2& params, nvinfer1::DataType dataType, int32_t numSegments,
    int32_t maxSeqLen, int32_t numHeads, int32_t headSize, void* ropedQkv, int32_t const* cuSeqLens, void* output)
{
    (void) dataType;
    setAlpha(params.scale_bmm1, 1.0F / std::sqrt(static_cast<float>(headSize)));
    setAlpha(params.scale_softmax, 1.0F);
    setAlpha(params.scale_bmm2, 1.0F);

    params.qkv_ptr = ropedQkv;
    params.o_ptr = output;
    params.cu_q_seqlens = const_cast<int32_t*>(cuSeqLens);
    params.cu_kv_seqlens = const_cast<int32_t*>(cuSeqLens);
    params.b = numSegments;
    params.h = numHeads;
    params.h_kv = numHeads;
    params.h_q_per_kv = 1;
    params.s = maxSeqLen;
    params.s_kv = maxSeqLen;
    params.d = headSize;
    params.dv = headSize;
    params.is_s_padded = true;

    int64_t const qkvStrideInBytes = static_cast<int64_t>(3) * numHeads * headSize * sizeof(half);
    params.q_stride_in_bytes = qkvStrideInBytes;
    params.k_stride_in_bytes = qkvStrideInBytes;
    params.v_stride_in_bytes = qkvStrideInBytes;
    params.o_stride_in_bytes = numHeads * headSize * sizeof(half);
}

void dispatchViTFMHA(nvinfer1::DataType dataType, int32_t smVersion, int32_t numSegments, int32_t maxSeqLen,
    int32_t numHeads, int32_t headSize, FusedMultiheadAttentionParamsV2& params, cudaStream_t stream)
{
    bool const useTiled = maxSeqLen > 64 && headSize >= 128;
    FMHAKernelHashKey key{trtToFMHADataType(dataType), maxSeqLen, headSize, true, false, true, 0, useTiled, 0};
    auto* kernelList = ViTFMHAKernelLoader::instance().get(trtToFMHADataType(dataType), smVersion);
    FMHAKernelFuncInfo kernelInfo = kernelList->find(key);
    if (kernelInfo.sharedMemBytes == 0)
    {
        throw std::runtime_error(describeFMHAKernelKey(key, smVersion));
    }

    void* kernelParams[] = {&params, nullptr};
    int32_t const unroll = (maxSeqLen + kernelInfo.unrollStep - 1) / kernelInfo.unrollStep;
    CUDA_DRIVER_CHECK(cuLaunchKernel(kernelInfo.deviceFunction, unroll, numHeads, numSegments, kernelInfo.threadsPerCTA,
        1, 1, kernelInfo.sharedMemBytes, stream, kernelParams, nullptr));
}

} // namespace

ViTAttentionRunner::ViTAttentionRunner(
    nvinfer1::DataType dataType, int32_t batchSize, int32_t seqLen, int32_t maxSeqLen, int32_t numHeads,
    int32_t headSize, int32_t maskRows, int32_t maskBlockSize, ViTAttentionMaskType maskType)
    : mDataType(dataType)
    , mBatchSize(batchSize)
    , mSeqLen(seqLen)
    , mMaxSeqLen(maxSeqLen > 0 ? maxSeqLen : seqLen)
    , mNumHeads(numHeads)
    , mHeadSize(headSize)
    , mMaskRows(maskRows)
    , mMaskBlockSize(maskBlockSize)
    , mMaskType(maskType)
{
}

bool ViTAttentionRunner::canImplement(nvinfer1::DataType dataType, int32_t numHeads, int32_t headSize)
{
    bool const typeSupported = dataType == nvinfer1::DataType::kHALF || dataType == nvinfer1::DataType::kFLOAT;
    return typeSupported && numHeads > 0 && headSize > 0;
}

bool ViTAttentionRunner::canImplementFMHA(nvinfer1::DataType dataType, int32_t headSize)
{
    // Qwen2.5-VL's vision tower has hidden_size=1280 and 16 heads, so its
    // ViT attention head size is 80. The Qwen LLM head size is separately 128.
    return dataType == nvinfer1::DataType::kHALF && (headSize == 64 || headSize == 80 || headSize == 128);
}

size_t ViTAttentionRunner::getWorkspaceSize(nvinfer1::DataType dataType, int32_t batchSize, int32_t seqLen,
    int32_t numHeads, int32_t headSize, ViTAttentionMaskType maskType)
{
    (void) dataType;
    if (batchSize <= 0 || seqLen <= 0 || numHeads <= 0 || headSize <= 0)
    {
        return 0;
    }
    if (maskType == ViTAttentionMaskType::kPackedCuSeqLens)
    {
        return static_cast<size_t>(batchSize) * seqLen * 3 * numHeads * headSize * sizeof(half);
    }
    size_t const ropeQKElems = static_cast<size_t>(batchSize) * numHeads * seqLen * headSize * 2;
    return ropeQKElems * sizeof(float);
}

void ViTAttentionRunner::dispatch(
    void const* qkv, void const* cos, void const* sin, void const* maskOrCuSeqLens, void* output, void* workspace,
    cudaStream_t stream) const
{
    check::check(qkv != nullptr, "ViTAttentionRunner requires a non-null QKV input.");
    check::check(cos != nullptr, "ViTAttentionRunner requires a non-null RoPE cosine input.");
    check::check(sin != nullptr, "ViTAttentionRunner requires a non-null RoPE sine input.");
    check::check(maskOrCuSeqLens != nullptr, "ViTAttentionRunner requires a non-null mask input.");
    check::check(output != nullptr, "ViTAttentionRunner requires a non-null output.");
    check::check(workspace != nullptr, "ViTAttentionRunner requires a non-null softmax workspace.");
    check::check(canImplement(mDataType, mNumHeads, mHeadSize), "Unsupported ViT attention configuration.");

    if (mMaskType == ViTAttentionMaskType::kPackedCuSeqLens)
    {
        check::check(canImplementFMHA(mDataType, mHeadSize), "Unsupported ViT FMHA configuration.");
        int32_t const numSegments = mMaskRows - 1;
        check::check(numSegments > 0, "ViT FMHA requires at least one cu_seqlens segment.");

        void* ropedQkv = workspace;
        kernel::launchBuildRopedPackedQKV(
            mDataType, qkv, cos, sin, ropedQkv, mBatchSize, mSeqLen, mNumHeads, mHeadSize, stream);
        CUDA_CHECK(cudaPeekAtLastError());

        FusedMultiheadAttentionParamsV2 params{};
        setupFMHAParams(params, mDataType, numSegments, mMaxSeqLen, mNumHeads, mHeadSize, ropedQkv,
            static_cast<int32_t const*>(maskOrCuSeqLens), output);
        dispatchViTFMHA(
            mDataType, getCurrentSMVersion(), numSegments, mMaxSeqLen, mNumHeads, mHeadSize, params, stream);
        return;
    }

    float* softmaxWorkspace = static_cast<float*>(workspace);
    kernel::launchViTAttention(mDataType, qkv, cos, sin, maskOrCuSeqLens, output, softmaxWorkspace,
        mBatchSize, mSeqLen, mNumHeads, mHeadSize, mMaskRows, mMaskBlockSize, mMaskType, stream);
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace trt_edgellm
