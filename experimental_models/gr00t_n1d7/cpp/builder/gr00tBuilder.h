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

#pragma once

#include <NvInfer.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace gr00t
{

//! The policy components, in dependency order.
std::vector<std::string> const& policyComponents();

//! \brief True when the selected export components carry one ONNX-route identity and every
//! component already staged in \p engineRoot that this build does not replace carries the same.
bool verifyGr00tExportIds(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components);

//! \brief Build the requested components from an exported ONNX root into the ONNX-less bundle layout.
//! \param onnxRoot  Export root holding one subdirectory per component plus ``tokenizer/``.
//! \param engineRoot  Output root; each component lands in ``<engineRoot>/<component>``.
//! \param components  Components to build (defaults to all via policyComponents()).
//! \param maxBatchSize  Widens the batch axis of every profiled input.
bool buildGr00tPolicy(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components, int32_t maxBatchSize);

//! Builds ``vl_backbone`` or ``action`` from its exported ONNX contract. The
//! optimization profile comes entirely from the component's config.json; the
//! ``visual`` tower goes through the shared VisualBuilder instead.
class Gr00tBuilder
{
public:
    Gr00tBuilder(std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir,
        std::string const& component, int32_t maxBatchSize);

    bool build();

private:
    bool parseConfig();
    bool setupOptimizationProfile(nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config,
        nvinfer1::INetworkDefinition const& network) const;
    bool inputExists(nvinfer1::INetworkDefinition const& network, std::string const& name) const;

    std::filesystem::path mOnnxDir;
    std::filesystem::path mEngineDir;
    std::string mComponent;
    int32_t mMaxBatchSize{1};
    std::string mOnnxFilename{"model.onnx"};
    std::string mEngineFilename;
    nlohmann::json mModelConfig;
};

} // namespace gr00t
} // namespace trt_edgellm
