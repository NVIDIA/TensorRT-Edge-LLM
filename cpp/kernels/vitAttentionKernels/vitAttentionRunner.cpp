/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vitAttentionRunner.h"

#include "common/checkMacros.h"

#include <cstddef>
#include <string>

namespace trt_edgellm
{

ViTAttentionRunner::ViTAttentionRunner(
    nvinfer1::DataType dataType, int32_t batchSize, int32_t seqLen, int32_t numHeads, int32_t headSize,
    int32_t maskRows)
    : mDataType(dataType)
    , mBatchSize(batchSize)
    , mSeqLen(seqLen)
    , mNumHeads(numHeads)
    , mHeadSize(headSize)
    , mMaskRows(maskRows)
{
}

bool ViTAttentionRunner::canImplement(nvinfer1::DataType dataType, int32_t numHeads, int32_t headSize)
{
    bool const typeSupported = dataType == nvinfer1::DataType::kHALF || dataType == nvinfer1::DataType::kFLOAT;
    return typeSupported && numHeads > 0 && headSize > 0;
}

size_t ViTAttentionRunner::getWorkspaceSize(int32_t batchSize, int32_t seqLen, int32_t numHeads, int32_t headSize)
{
    if (batchSize <= 0 || seqLen <= 0 || numHeads <= 0 || headSize <= 0)
    {
        return 0;
    }
    size_t const ropeQKElems = static_cast<size_t>(batchSize) * numHeads * seqLen * headSize * 2;
    return ropeQKElems * sizeof(float);
}

void ViTAttentionRunner::dispatch(
    void const* qkv, void const* cos, void const* sin, void const* attentionMask, void* output, void* workspace,
    cudaStream_t stream) const
{
    check::check(qkv != nullptr, "ViTAttentionRunner requires a non-null QKV input.");
    check::check(cos != nullptr, "ViTAttentionRunner requires a non-null RoPE cosine input.");
    check::check(sin != nullptr, "ViTAttentionRunner requires a non-null RoPE sine input.");
    check::check(attentionMask != nullptr, "ViTAttentionRunner requires a non-null attention mask input.");
    check::check(output != nullptr, "ViTAttentionRunner requires a non-null output.");
    check::check(workspace != nullptr, "ViTAttentionRunner requires a non-null softmax workspace.");
    check::check(canImplement(mDataType, mNumHeads, mHeadSize), "Unsupported ViT attention configuration.");

    kernel::launchViTAttention(mDataType, qkv, cos, sin, attentionMask, output, static_cast<float*>(workspace),
        mBatchSize, mSeqLen, mNumHeads, mHeadSize, mMaskRows, stream);
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace trt_edgellm
