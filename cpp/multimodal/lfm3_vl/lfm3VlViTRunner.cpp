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

#include "multimodal/lfm3_vl/lfm3VlViTRunner.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"

namespace trt_edgellm
{
namespace rt
{

namespace
{
constexpr int64_t kCONV_TAPS = 9;
} // namespace

bool Lfm3VlViTRunner::validateExtraConfig(nlohmann::json const& jsonConfig)
{
    if (!jsonConfig.contains("vision_config") || !jsonConfig["vision_config"].is_object())
    {
        LOG_ERROR("LFM3-VL requires a vision_config object in config.json");
        return false;
    }
    auto const& visionConfig = jsonConfig["vision_config"];
    if (!visionConfig.contains("rope_parameters") || !visionConfig["rope_parameters"].is_object()
        || !visionConfig["rope_parameters"].contains("rope_theta"))
    {
        LOG_ERROR("LFM3-VL requires vision_config.rope_parameters.rope_theta in config.json");
        return false;
    }
    mVitRopeTheta = visionConfig["rope_parameters"]["rope_theta"].get<float>();
    if (mVitRopeTheta <= 0.0F)
    {
        LOG_ERROR("LFM3-VL vision rope_theta must be positive, got %f", mVitRopeTheta);
        return false;
    }
    return true;
}

bool Lfm3VlViTRunner::usesTextMRope() const
{
    return false;
}

void Lfm3VlViTRunner::buildRotaryPosEmb(std::vector<VisionSpan> const& spans, cudaStream_t stream)
{
    for (auto const& s : spans)
    {
        kernel::initRotaryPosEmbQwenViT(mRotaryPosEmb, {s.vit.gridT, s.vit.gridH, s.vit.gridW}, mConfig.mergeSize,
            s.vit.patchStart, mVitRopeTheta, 1.0F, stream);
    }
}

bool Lfm3VlViTRunner::allocateExtraBuffers(int64_t /*maxImageTokens*/)
{
    mConvNeighborIndex = rt::Tensor({mConfig.maxHW, kCONV_TAPS}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "Lfm3VlViTRunner::mConvNeighborIndex");
    return mVisualContext->setTensorAddress(binding_names::kConvNeighborIndex, mConvNeighborIndex.rawPointer());
}

void Lfm3VlViTRunner::buildExtraInputs(
    std::vector<VisionSpan> const& spans, int64_t totalSeqLength, int64_t /*totalImageTokens*/, cudaStream_t stream)
{
    check::check(mConvNeighborIndex.reshape({totalSeqLength, kCONV_TAPS}), "Tensor reshape failed");
    for (auto const& s : spans)
    {
        kernel::initConvNeighborIndexLfm3VlViT(
            mConvNeighborIndex, {s.vit.gridT, s.vit.gridH, s.vit.gridW}, mConfig.mergeSize, s.vit.patchStart, stream);
    }
}

bool Lfm3VlViTRunner::bindExtraInputShapes()
{
    return mVisualContext->setInputShape(binding_names::kConvNeighborIndex, mConvNeighborIndex.getShape().getTRTDims());
}

} // namespace rt
} // namespace trt_edgellm
