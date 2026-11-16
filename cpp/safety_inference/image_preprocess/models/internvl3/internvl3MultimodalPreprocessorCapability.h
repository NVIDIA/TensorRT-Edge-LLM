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

#pragma once

#include "internvlImagePlan.h"
#include "safety_inference/image_preprocess/multimodalPreprocessorCapability.h"

namespace trt_edgellm::safety
{

class InternVL3MultimodalPreprocessorCapability final : public MultimodalPreprocessorCapability
{
public:
    SafetyStatus declareResources(
        ImageProcessingLimits const& limits, ResourceDeclaration& declaration) noexcept override;
    SafetyStatus initialize(PreparedResources const& prepared, LMStageContext const& visual) noexcept override;
    ImageProcessingPlan planImage(vlm::ImageInputView image) noexcept override;
    SafetyStatus prepareVisualInput(LMStageContext const& context, vlm::ImageInputView image) noexcept override;
    SafetyStatus deinitialize() noexcept override;

private:
    ImageProcessingLimits mLimits{};
    RuntimeMemoryView mPinnedRgb{};
    RuntimeMemoryView mDeviceRgb{};
    RuntimeMemoryView mTileScratch{};
    RuntimeTensorView const* mVisualInput{nullptr};
    cudaStream_t mStream{nullptr};
    LMStageContext const* mVisualContext{nullptr};
    InternVLTilePlan mPlan{};
    vlm::ImageInputView mPlannedImage{};
    CapabilityPreparationState mState{CapabilityPreparationState::kEMPTY};
    bool mPlanValid{false};
};

} // namespace trt_edgellm::safety
