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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "builder/safety/safetyBuilderConfig.h"

#include <NvInfer.h>
#include <filesystem>

namespace trt_edgellm::safety::builder
{

class SafetyVisualBuilder
{
public:
    SafetyVisualBuilder(
        std::filesystem::path onnxDir, std::filesystem::path engineDir, SafetyVisualBuilderConfig config);

    bool build() noexcept;

private:
    bool parseConfig() noexcept;
    bool setupOptimizationProfiles(
        nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network);
    bool validateOutputTokenShape(nvinfer1::INetworkDefinition& network);
    bool saveArtifacts(std::filesystem::path const& directory);

    std::filesystem::path mOnnxDir;
    std::filesystem::path mEngineDir;
    SafetyVisualBuilderConfig mBuilderConfig;
    int64_t mNumChannels{0};
    int64_t mImageSizeH{0};
    int64_t mImageSizeW{0};
    int64_t mTokensPerTile{0};
    Json mModelConfig;
};

} // namespace trt_edgellm::safety::builder
