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
namespace pi05
{

//! The policy components \p onnxRoot holds, in dependency order.
std::vector<std::string> policyComponents(std::filesystem::path const& onnxRoot);

//! Verify that every selected source component names one export and that \p engineRoot
//! is empty or already holds only that export. A root holding another export cannot be
//! rebuilt in place: its untagged assets/ and text_tokenizer/ would survive unnoticed.
bool verifyPi05ExportIds(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components);

//! Build the requested policy components from a package-exported ONNX root and
//! stage the runtime sidecars (token-embedding table, tokenizer, normalization
//! assets) next to the engines, so the engine directory is self-contained.
//!
//! \param onnxRoot  Export root holding one subdirectory per component.
//! \param engineRoot  Output root; each component lands in ``<engineRoot>/<component>``.
//! \param components  Components to build (defaults to all via policyComponents()).
//! \param maxBatchSize  Widens the batch axis of every input declared in the contract.
bool buildPi05Policy(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components, int32_t maxBatchSize);

//! Builds one pi0.5 component from a package-exported ONNX contract. The optimization
//! profile comes entirely from the component's config.json, so this class holds no
//! model-specific tensor names.
class Pi05Builder
{
public:
    Pi05Builder(std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir,
        std::string const& component, int32_t maxBatchSize);

    bool build();

private:
    bool parseConfig();
    bool setupOptimizationProfile(nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config,
        nvinfer1::INetworkDefinition const& network) const;
    //! Reject a profile entry naming a tensor the network does not declare.
    //! Cosmos3 warns and skips; a silently dropped profile yields an engine
    //! whose dynamic range differs from the exported contract.
    bool validateProfileNames(nvinfer1::INetworkDefinition const& network) const;
    bool inputExists(nvinfer1::INetworkDefinition const& network, std::string const& name) const;

    std::filesystem::path mOnnxDir;
    std::filesystem::path mEngineDir;
    std::string mComponent;
    int32_t mMaxBatchSize{1};
    std::string mOnnxFilename{"model.onnx"};
    std::string mEngineFilename;

    nlohmann::json mModelConfig;
};

} // namespace pi05
} // namespace trt_edgellm
