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

#include "internvl3MultimodalPreprocessorCapability.h"

#include "internvl3PreprocessKernels.cuh"

#include "safety_inference/runtime/safetyStatusUtils.h"
#include "safety_inference/trt_runner/lmStageContext.h"

#include <cstring>
#include <limits>

namespace trt_edgellm::safety
{
namespace
{
constexpr RuntimeResourceKey kPINNED_RGB{200};
constexpr RuntimeResourceKey kDEVICE_RGB{201};
constexpr RuntimeResourceKey kTILE_SCRATCH{202};
constexpr RuntimeResourceKey kVISUAL_STREAM{203};
constexpr RuntimeResourceKey kVISUAL_INPUT{204};

SafetyStatus invalid(char const* message) noexcept
{
    return SafetyStatus::error(SafetyStatusCode::kInvalidArgument, message);
}

bool multiply(std::size_t left, std::size_t right, std::size_t& output) noexcept
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
    {
        return false;
    }
    output = left * right;
    return true;
}

RgbNormalization normalization(ImageProcessingLimits const& limits) noexcept
{
    return {limits.normalizationScale,
        {limits.normalizationMean[0], limits.normalizationMean[1], limits.normalizationMean[2]},
        {limits.normalizationDeviation[0], limits.normalizationDeviation[1], limits.normalizationDeviation[2]}, 0.5F};
}
} // namespace

SafetyStatus InternVL3MultimodalPreprocessorCapability::declareResources(
    ImageProcessingLimits const& limits, ResourceDeclaration& declaration) noexcept
{
    if (mState != CapabilityPreparationState::kEMPTY || limits.maxImageWidth <= 0 || limits.maxImageHeight <= 0
        || limits.maxImageWidth > std::numeric_limits<int32_t>::max() / kInternVLChannels || limits.maxMainBlocks <= 0
        || limits.maxMainBlocks > kMaxInternVLTiles || limits.minBlocks <= 0 || limits.minBlocks > limits.maxMainBlocks
        || limits.visualProfileBlocks <= 0 || limits.visualProfileBlocks > kMaxInternVLTiles
        || limits.tokensPerTile <= 0 || limits.normalizationDeviation[0] <= 0.0F
        || limits.normalizationDeviation[1] <= 0.0F || limits.normalizationDeviation[2] <= 0.0F)
    {
        return invalid("Invalid InternVL image limits");
    }
    std::size_t rgbBytes{0};
    std::size_t scratchBytes{0};
    int32_t const thumbnail = limits.addThumbnail && limits.maxMainBlocks > 1 ? 1 : 0;
    if (!multiply(static_cast<std::size_t>(limits.maxImageWidth), kInternVLChannels, rgbBytes)
        || !multiply(rgbBytes, static_cast<std::size_t>(limits.maxImageHeight), rgbBytes)
        || !multiply(static_cast<std::size_t>(limits.maxMainBlocks + thumbnail), kInternVLTileValues, scratchBytes)
        || !multiply(scratchBytes, sizeof(half), scratchBytes)
        || limits.maxMainBlocks + thumbnail > limits.visualProfileBlocks)
    {
        return invalid("InternVL image resource size exceeds its bound");
    }
    try
    {
        declaration.memory.push_back({kPINNED_RGB, rgbBytes, RuntimeMemoryKind::kPINNED});
        declaration.memory.push_back({kDEVICE_RGB, rgbBytes, RuntimeMemoryKind::kDEVICE});
        declaration.memory.push_back({kTILE_SCRATCH, scratchBytes, RuntimeMemoryKind::kDEVICE});
        declaration.borrowed.push_back(
            {kVISUAL_STREAM, RuntimeBorrowedResourceKind::kCUDA_STREAM, "visual_encode", 0, {}});
        declaration.borrowed.push_back(
            {kVISUAL_INPUT, RuntimeBorrowedResourceKind::kIO_BINDING, "visual_encode", 0, "input"});
    }
    catch (...)
    {
        return SafetyStatus::error(SafetyStatusCode::kFailedAllocation, "Cannot declare InternVL image resources");
    }
    mLimits = limits;
    mState = CapabilityPreparationState::kDECLARED;
    return SafetyStatus::ok();
}

SafetyStatus InternVL3MultimodalPreprocessorCapability::initialize(
    PreparedResources const& prepared, LMStageContext const& visual) noexcept
{
    if (mState != CapabilityPreparationState::kDECLARED || visual.plan.id != "visual_encode"
        || visual.mainStream == nullptr || visual.profiles.size() != 1)
    {
        return invalid("Invalid InternVL visual context");
    }
    auto status = prepared.memory(kPINNED_RGB, mPinnedRgb);
    if (status.okStatus())
    {
        status = prepared.memory(kDEVICE_RGB, mDeviceRgb);
    }
    if (status.okStatus())
    {
        status = prepared.memory(kTILE_SCRATCH, mTileScratch);
    }
    if (status.okStatus())
    {
        status = prepared.stream(kVISUAL_STREAM, mStream);
    }
    if (status.okStatus())
    {
        status = prepared.tensor(kVISUAL_INPUT, mVisualInput);
    }
    if (!status.okStatus())
    {
        return status;
    }
    auto const* profile = mVisualInput->desc != nullptr && mVisualInput->desc->profiles.size() == 1
        ? &mVisualInput->desc->profiles.front()
        : nullptr;
    if (mStream != visual.mainStream || mVisualInput->memory.kind != RuntimeMemoryKind::kDEVICE || profile == nullptr
        || mVisualInput->desc->dataType != nvinfer1::DataType::kHALF
        || mVisualInput->desc->ioMode != nvinfer1::TensorIOMode::kINPUT || profile->shape.nbDims != 4
        || profile->shape.d[0] != mLimits.visualProfileBlocks || profile->shape.d[1] != kInternVLChannels
        || profile->shape.d[2] != kInternVLTileSize || profile->shape.d[3] != kInternVLTileSize)
    {
        return invalid("Visual input binding is incompatible with InternVL preprocessing");
    }
    mVisualContext = &visual;
    mPlanValid = false;
    mState = CapabilityPreparationState::kREADY;
    return SafetyStatus::ok();
}

ImageProcessingPlan InternVL3MultimodalPreprocessorCapability::planImage(vlm::ImageInputView image) noexcept
{
    if (mState != CapabilityPreparationState::kREADY || image.data == nullptr || image.width <= 0 || image.height <= 0
        || image.width > mLimits.maxImageWidth || image.height > mLimits.maxImageHeight
        || static_cast<std::size_t>(image.width) > std::numeric_limits<std::size_t>::max() / kInternVLChannels)
    {
        return {invalid("Invalid InternVL RGB image"), 0};
    }
    std::size_t const rowBytes = static_cast<std::size_t>(image.width) * kInternVLChannels;
    if (image.rowStride < rowBytes || static_cast<std::size_t>(image.height) > image.sizeInBytes / image.rowStride
        || !computeInternVLTilePlan(
            image.width, image.height, mLimits.minBlocks, mLimits.maxMainBlocks, mLimits.addThumbnail, mPlan))
    {
        return {invalid("InternVL image extent or tile plan is out of range"), 0};
    }
    if (static_cast<std::size_t>(mPlan.totalTiles) > mLimits.maxImageTokens / mLimits.tokensPerTile
        || mPlan.totalTiles != mLimits.visualProfileBlocks)
    {
        return {invalid("InternVL image plan exceeds the fixed profile"), 0};
    }
    mPlan.imageTokenCount = static_cast<std::size_t>(mPlan.totalTiles) * mLimits.tokensPerTile;
    mPlannedImage = image;
    mPlanValid = true;
    return {SafetyStatus::ok(), mPlan.imageTokenCount};
}

SafetyStatus InternVL3MultimodalPreprocessorCapability::prepareVisualInput(
    LMStageContext const& context, vlm::ImageInputView image) noexcept
{
    if (mState != CapabilityPreparationState::kREADY || !mPlanValid || &context != mVisualContext
        || context.mainStream != mStream || image.data != mPlannedImage.data || image.width != mPlannedImage.width
        || image.height != mPlannedImage.height || image.rowStride != mPlannedImage.rowStride)
    {
        return SafetyStatus::fatal(SafetyStatusCode::kInvalidState, "InternVL image plan does not match the request");
    }
    std::size_t const rowBytes = static_cast<std::size_t>(image.width) * kInternVLChannels;
    std::size_t const packedBytes = rowBytes * static_cast<std::size_t>(image.height);
    if (packedBytes > mPinnedRgb.sizeInBytes || packedBytes > mDeviceRgb.sizeInBytes)
    {
        return invalid("InternVL image exceeds prepared storage");
    }
    for (int32_t row = 0; row < image.height; ++row)
    {
        auto const* source = static_cast<uint8_t const*>(image.data) + static_cast<std::size_t>(row) * image.rowStride;
        auto* destination = static_cast<uint8_t*>(mPinnedRgb.data) + static_cast<std::size_t>(row) * rowBytes;
        std::memcpy(destination, source, rowBytes);
    }
    auto cudaStatus = cudaMemcpyAsync(mDeviceRgb.data, mPinnedRgb.data, packedBytes, cudaMemcpyHostToDevice, mStream);
    if (cudaStatus == cudaSuccess)
    {
        cudaStatus = launchInternVLPreprocess(static_cast<uint8_t const*>(mDeviceRgb.data), image.width, image.height,
            static_cast<int32_t>(rowBytes), mPlan, normalization(mLimits), static_cast<half*>(mTileScratch.data),
            static_cast<half*>(mVisualInput->memory.data), mStream);
    }
    return cudaStatus == cudaSuccess ? SafetyStatus::ok() : fromCudaError(cudaStatus, "InternVL image preprocessing");
}

SafetyStatus InternVL3MultimodalPreprocessorCapability::deinitialize() noexcept
{
    if (mState == CapabilityPreparationState::kEMPTY || mState == CapabilityPreparationState::kCLOSED)
    {
        return SafetyStatus::ok();
    }
    mPinnedRgb = {};
    mDeviceRgb = {};
    mTileScratch = {};
    mVisualInput = nullptr;
    mStream = nullptr;
    mVisualContext = nullptr;
    mPlan = {};
    mPlannedImage = {};
    mPlanValid = false;
    mState = CapabilityPreparationState::kCLOSED;
    return SafetyStatus::ok();
}

} // namespace trt_edgellm::safety
