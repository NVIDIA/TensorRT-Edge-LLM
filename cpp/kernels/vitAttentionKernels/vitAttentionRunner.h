/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

namespace trt_edgellm
{

class ViTAttentionRunner
{
public:
    ViTAttentionRunner(nvinfer1::DataType dataType, int32_t batchSize, int32_t seqLen, int32_t numHeads,
        int32_t headSize, int32_t maskRows);

    static bool canImplement(nvinfer1::DataType dataType, int32_t numHeads, int32_t headSize);
    static size_t getWorkspaceSize(int32_t batchSize, int32_t seqLen, int32_t numHeads, int32_t headSize);

    void dispatch(
        void const* qkv, void const* cos, void const* sin, void const* attentionMask, void* output, void* workspace,
        cudaStream_t stream) const;

private:
    nvinfer1::DataType mDataType;
    int32_t mBatchSize;
    int32_t mSeqLen;
    int32_t mNumHeads;
    int32_t mHeadSize;
    int32_t mMaskRows;
};

namespace kernel
{

void launchViTAttention(nvinfer1::DataType dataType, void const* qkv, void const* cos, void const* sin,
    void const* attentionMask, void* output, float* softmaxWorkspace, int32_t batchSize, int32_t seqLen,
    int32_t numHeads, int32_t headSize, int32_t maskRows, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
