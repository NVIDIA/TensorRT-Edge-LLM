/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace trt_edgellm
{

enum class ViTAttentionMaskType : int32_t
{
    kDenseAdditive = 0,
    kPackedCuSeqLens = 1,
    kCompactBlock = 2,
};

class ViTAttentionRunner
{
public:
    ViTAttentionRunner(nvinfer1::DataType dataType, int32_t batchSize, int32_t seqLen, int32_t maxSeqLen,
        int32_t numHeads, int32_t headSize, int32_t maskRows, int32_t maskBlockSize, ViTAttentionMaskType maskType);

    static bool canImplement(nvinfer1::DataType dataType, int32_t numHeads, int32_t headSize);
    static bool canImplementFMHA(nvinfer1::DataType dataType, int32_t headSize);
    static size_t getWorkspaceSize(nvinfer1::DataType dataType, int32_t batchSize, int32_t seqLen, int32_t numHeads,
        int32_t headSize, ViTAttentionMaskType maskType);

    void dispatch(
        void const* qkv, void const* cos, void const* sin, void const* maskOrCuSeqLens, void* output,
        void* workspace, cudaStream_t stream) const;

private:
    nvinfer1::DataType mDataType;
    int32_t mBatchSize;
    int32_t mSeqLen;
    int32_t mMaxSeqLen;
    int32_t mNumHeads;
    int32_t mHeadSize;
    int32_t mMaskRows;
    int32_t mMaskBlockSize;
    ViTAttentionMaskType mMaskType;
};

namespace kernel
{

void launchViTAttention(nvinfer1::DataType dataType, void const* qkv, void const* cos, void const* sin,
    void const* attentionMask, void* output, float* softmaxWorkspace, int32_t batchSize, int32_t seqLen,
    int32_t numHeads, int32_t headSize, int32_t maskRows, int32_t maskBlockSize, ViTAttentionMaskType maskType,
    cudaStream_t stream);

void launchBuildRopedPackedQKV(nvinfer1::DataType dataType, void const* qkv, void const* cos, void const* sin,
    void* ropedQkv, int32_t batchSize, int32_t seqLen, int32_t numHeads, int32_t headSize, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
